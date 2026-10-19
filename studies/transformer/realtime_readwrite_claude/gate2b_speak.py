"""Gate 2b (PREREG_GATE2B.md): the stored value is the model's own "about to say it" state.

For each fact "Ben lives in Oslo." the writer runs the rehearsal "Ben lives in Oslo. Ben lives in" and keeps
the last-token state. At question time the two-hop memory retrieves that state and adds it to the stream at
the last token of "Q: Where does Anna's brother live?\nA: Anna's brother lives in"; the model answers itself.

    python gate2b_speak.py --model Qwen/Qwen2.5-0.5B --out results/gate2b_qwen2.5-0.5b.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

import gate1_twohop as g1
import gate2_speak as g2

N = g2.N
VLAYERS = (10, 14, 17, 20, 22)
GRID = list(itertools.product(VLAYERS, (1.0, 2.0, 4.0, 8.0)))
HOP1 = ("outer", 17, 32, False)  # fixed from Gate 1b's dev selection
ORACLE_MIN, ORACLE_TOP1_MIN, SPEAK_RATIO, CONTROL_MAX = 0.80, 0.50, 0.85, 0.25


def rehearsal(w, i):
    return f"{w.targets[i]} lives in {w.cities[i]}. {w.targets[i]} lives in"


def collect(model, encode_offsets, pad_id, device, seeds, names, answers, batch):
    out = []
    for s in seeds:
        c = g2.collect_world(model, encode_offsets, pad_id, device, g2.make_world(s, names), batch, answers)
        texts = [rehearsal(c["world"], i) for i in range(N)]
        R, Z = g1.run_texts(model, encode_offsets, pad_id, device, texts, [[len(t)] for t in texts], VLAYERS,
                            batch, want_logits=True)
        c["R"] = np.stack([r[0] for r in R])  # [N, len(VLAYERS), d]
        c["rehearsal_logits"] = Z
        out.append(c)
    return out


def write_vectors(c, kb, mu, Lv, kind):
    w, S = c["world"], c["S"]
    vals = c["R"][:, VLAYERS.index(Lv)] - mu[Lv]
    if kind == "memory":
        M1, _, _, kf = g2.memories(c, kb)
        M3 = g1.Delta(kf.shape[1], vals.shape[1])
        for k, i in w.order:
            if k == "fact":
                M3.write(kf[i], vals[i])
        kq = kb.hop1(S, "query")
        return g1._unit(np.stack([M3.read(kb.clean(M1.read(kq[i]))) for i in range(N)]))
    if kind == "oracle":
        return g1._unit(vals)
    if kind == "wrong":
        return g1._unit(vals[g2.derangement(30_000 + w.seed)])
    raise ValueError(kind)


def speak(model, encode_offsets, pad_id, device, first_id, worlds, kb, mu, Lv, alpha, kind, batch):
    w8 = t1 = 0
    for c in worlds:
        vecs = None if kind == "none" else write_vectors(c, kb, mu, Lv, kind)
        texts = [g2.answer_prompt(c["world"], i) for i in range(N)]
        Z = g2.inject_logits(model, encode_offsets, pad_id, device, texts, vecs, Lv, alpha, batch)
        a8, a1 = g2.score_logits(Z, c["world"], first_id)
        w8, t1 = w8 + a8, t1 + a1
    m = len(worlds) * N
    return w8 / m, t1 / m


def run(model, encode_offsets, first_id, pad_id, device, fit_seeds=g2.FIT_SEEDS, select_seeds=g2.SELECT_SEEDS,
        test_seeds=g2.TEST_SEEDS, grid=GRID, hop1=HOP1, batch=8, log=print):
    t0 = time.time()
    fit = collect(model, encode_offsets, pad_id, device, fit_seeds, g1.DEV_NAMES, False, batch)
    sel = collect(model, encode_offsets, pad_id, device, select_seeds, g1.DEV_NAMES, False, batch)
    test = collect(model, encode_offsets, pad_id, device, test_seeds, g1.TEST_NAMES, True, batch)
    log(f"states collected ({time.time() - t0:.0f}s)")
    dev = fit + sel
    n = len(test) * N
    mean = lambda ws: {L: np.mean(np.concatenate([c["R"][:, VLAYERS.index(L)] for c in ws]), 0) for L in VLAYERS}
    mu_fit, mu_dev = mean(fit), mean(dev)
    kb_fit, kb = g2.KeyBuilder(fit, *hop1), g2.KeyBuilder(dev, *hop1)

    two_hop = g2.acc([g2.eval_1b(c, kb) for c in test], "two_hop")
    no_write = sum(g2.score_logits(c["no_write_logits"], c["world"], first_id)[0] for c in test) / n
    in_ctx = sum(g2.score_logits(c["in_context_logits"], c["world"], first_id)[0] for c in test) / n
    rehearse_top1 = sum(g2.score_logits(c["rehearsal_logits"], c["world"], first_id)[1] for c in test) / n

    sel_rows = []
    for Lv, al in grid:
        a8, a1 = speak(model, encode_offsets, pad_id, device, first_id, sel, kb_fit, mu_fit, Lv, al, "memory", batch)
        sel_rows.append({"value_layer": Lv, "alpha": al, "select_way8": a8, "select_top1": a1})
    best = max(sel_rows, key=lambda r: r["select_way8"])
    Lv, al = best["value_layer"], best["alpha"]
    log(f"selection done ({time.time() - t0:.0f}s): layer {Lv}, alpha {al}")
    res = {k: speak(model, encode_offsets, pad_id, device, first_id, test, kb, mu_dev, Lv, al, k, batch)
           for k in ("memory", "oracle", "wrong")}
    out = {"hop1_config": list(hop1), "two_hop_same_keys": two_hop, "chosen": best, "selection": sel_rows,
           "memory_write_way8": res["memory"][0], "memory_write_top1": res["memory"][1],
           "oracle_write_way8": res["oracle"][0], "oracle_write_top1": res["oracle"][1],
           "wrong_write_right_city_way8": res["wrong"][0], "wrong_write_right_city_top1": res["wrong"][1],
           "no_write_way8": no_write, "in_context_ceiling": in_ctx, "rehearsal_copy_top1": rehearse_top1,
           "required_memory_write": SPEAK_RATIO * two_hop, "n_questions": n, "seconds": time.time() - t0}
    ok = (out["oracle_write_way8"] >= ORACLE_MIN and out["oracle_write_top1"] >= ORACLE_TOP1_MIN and
          out["memory_write_way8"] >= SPEAK_RATIO * two_hop and
          out["wrong_write_right_city_way8"] <= CONTROL_MAX and no_write <= CONTROL_MAX)
    out["decision"] = "PASS" if ok else "FAIL"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", default="results/gate2b_qwen2.5-0.5b.json")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def encode_offsets(s):
        e = tok(s, add_special_tokens=True, return_offsets_mapping=True)
        return e["input_ids"], e["offset_mapping"]

    first_id = lambda word: tok(" " + word, add_special_tokens=False)["input_ids"][0]
    r = run(model, encode_offsets, first_id, pad_id, device, batch=args.batch)
    r["model"] = args.model
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(r, f, indent=1)

    print(f"\n{args.model}   ({r['n_questions']} test questions, held-out names, chance 0.125)")
    print(f"  rehearsal check: after 'Ben lives in Oslo. Ben lives in' the model says the city {r['rehearsal_copy_top1']:.3f}")
    print(f"  two-hop memory with the fixed Gate 1b keys: {r['two_hop_same_keys']:.3f}")
    print(f"\nGATE 2b  chosen on dev: write at layer {r['chosen']['value_layer']}, alpha {r['chosen']['alpha']}")
    print("  the model answers itself, context empty:   8-way / full-vocab top-1")
    print(f"    memory write        {r['memory_write_way8']:.3f} / {r['memory_write_top1']:.3f}"
          f"   (8-way needs >= {r['required_memory_write']:.3f})")
    print(f"    oracle write        {r['oracle_write_way8']:.3f} / {r['oracle_write_top1']:.3f}"
          f"   (needs >= 0.80 / >= 0.50)")
    print(f"    wrong-city write    {r['wrong_write_right_city_way8']:.3f} / {r['wrong_write_right_city_top1']:.3f}"
          f"   (right-city rate, needs <= 0.25)")
    print(f"    no write            {r['no_write_way8']:.3f}   (needs <= 0.25)")
    print(f"    in-context ceiling  {r['in_context_ceiling']:.3f}")
    print(f"  DECISION: {r['decision']}")
    print("\n  dev selection (8-way / top-1):")
    for s in sorted(r["selection"], key=lambda s: -s["select_way8"]):
        print(f"    layer {s['value_layer']:2d} alpha {s['alpha']:3.1f}   {s['select_way8']:.3f} / {s['select_top1']:.3f}")


if __name__ == "__main__":
    main()
