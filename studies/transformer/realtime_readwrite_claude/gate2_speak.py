"""Gate 1b and Gate 2 (PREREG_GATE1B_2.md).

Gate 1b: hop 1 keyed on start + move (anchor state ⊕ phrase state), in worlds where every anchor and every
         relation appears twice, so neither part alone can resolve the binding.
Gate 2:  the retrieved city state is added back into the residual stream and the frozen model answers itself.

    python gate2_speak.py --model Qwen/Qwen2.5-0.5B --out results/gate2_qwen2.5-0.5b.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time
from dataclasses import dataclass

import numpy as np

import gate0_route_probe as g0
import gate1_twohop as g1

N, N_ANCH = 8, 4
FIT_SEEDS, SELECT_SEEDS, TEST_SEEDS = list(range(0, 30)), list(range(30, 50)), list(range(2000, 2100))
LAYERS = (6, 10, 14, 17)
GRID_1B = list(itertools.product(("concat", "outer"), (14, 17), (32, 64), (True, False)))
GRID_2 = list(itertools.product((6, 10, 14), (0.5, 1.0, 2.0, 4.0)))
EXPAND_N, TOPK = 2048, 32
PASS_ACC, CONTROL_MAX, ORACLE_MIN, SPEAK_RATIO = 0.80, 0.25, 0.80, 0.85


# ------------------------------------------------------------------------------------------- worlds

@dataclass
class World:
    seed: int
    anchors: list      # 4 names
    bind_anchor: list  # anchor index of binding i
    bind_rel: list     # relation of binding i
    targets: list      # 8 names, target of binding i
    cities: list       # cities[i] is where targets[i] lives
    chunk: str
    anchor_end: list
    rel_end: list
    bind_name_end: list
    fact_name_end: list
    fact_city_end: list
    order: list

    def ref(self, i):
        return f"{self.anchors[self.bind_anchor[i]]}'s {self.bind_rel[i]}"


def make_world(seed: int, names: list) -> World:
    rng = random.Random(seed)
    people = rng.sample(names, N_ANCH + N)
    anchors, targets = people[:N_ANCH], people[N_ANCH:]
    rels = rng.sample(g0.RELATIONS, N_ANCH)
    bind_anchor = [i // 2 for i in range(N)]
    bind_rel = [rels[(i // 2 + i % 2) % N_ANCH] for i in range(N)]
    cities = rng.sample(g1.CITIES, N)
    order = [("bind", i) for i in range(N)] + [("fact", i) for i in range(N)]
    rng.shuffle(order)
    text = ""
    a_end, r_end, b_end, f_end, c_end = ([0] * N for _ in range(5))
    for kind, i in order:
        if text:
            text += " "
        if kind == "bind":
            a = anchors[bind_anchor[i]]
            a_end[i] = len(text) + len(a)
            ref = f"{a}'s {bind_rel[i]}"
            r_end[i] = len(text) + len(ref)
            text += f"{ref} is {targets[i]}"
            b_end[i] = len(text)
            text += "."
        else:
            text += targets[i]
            f_end[i] = len(text)
            text += f" lives in {cities[i]}"
            c_end[i] = len(text)
            text += "."
    return World(seed, anchors, bind_anchor, bind_rel, targets, cities, text, a_end, r_end, b_end, f_end, c_end, order)


def question(w, i):
    a = w.anchors[w.bind_anchor[i]]
    pre = "Where does "
    return f"{pre}{w.ref(i)} live?", len(pre) + len(a), len(pre) + len(w.ref(i))


def answer_prompt(w, i, with_chunk=False):
    p = f"Q: Where does {w.ref(i)} live?\nA: {w.ref(i)} lives in"
    return (w.chunk + "\n" + p) if with_chunk else p


# ------------------------------------------------------------------------------------------- model I/O

def inject_logits(model, encode_offsets, pad_id, device, texts, vecs, layer, alpha, batch=8):
    """Last-position logits with alpha·‖h‖·v added to the residual stream at the last token after
    decoder layer `layer` (i.e. to hidden_states[layer]). vecs: [n, d] unit vectors, or None for no write."""
    import torch

    blocks = model.model.layers
    out = []
    for s in range(0, len(texts), batch):
        enc = [encode_offsets(t)[0] for t in texts[s:s + batch]]
        L = max(len(x) for x in enc)
        inp = torch.full((len(enc), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(enc), L), dtype=torch.long)
        for b, ids in enumerate(enc):
            inp[b, L - len(ids):] = torch.tensor(ids)
            mask[b, L - len(ids):] = 1
        pos = (mask.cumsum(dim=1) - 1).clamp(min=0)
        handle = None
        if vecs is not None and alpha != 0:
            V = torch.tensor(np.asarray(vecs[s:s + batch]), dtype=next(model.parameters()).dtype, device=device)

            def hook(mod, args, output, V=V):
                h = output[0] if isinstance(output, tuple) else output
                h = h.clone()
                last = h[:, -1, :]
                h[:, -1, :] = last + alpha * last.norm(dim=-1, keepdim=True) * V
                return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

            handle = blocks[layer - 1].register_forward_hook(hook)
        try:
            with torch.no_grad():
                o = model(input_ids=inp.to(device), attention_mask=mask.to(device), position_ids=pos.to(device))
        finally:
            if handle is not None:
                handle.remove()
        out.append(o.logits[:, -1].float().cpu().numpy())
    return np.concatenate(out)


def collect_world(model, encode_offsets, pad_id, device, w: World, batch, answers: bool):
    pos = w.anchor_end + w.rel_end + w.bind_name_end + w.fact_name_end + w.fact_city_end
    (C,), _ = g1.run_texts(model, encode_offsets, pad_id, device, [w.chunk], [pos], LAYERS, batch)
    qs = [question(w, i) for i in range(N)]
    Q, _ = g1.run_texts(model, encode_offsets, pad_id, device, [q for q, _, _ in qs],
                        [[a, r] for _, a, r in qs], LAYERS, batch)
    S = {"anchor_write": C[0:N], "rel_write": C[N:2 * N], "name_bind": C[2 * N:3 * N],
         "name_fact": C[3 * N:4 * N], "city_fact": C[4 * N:5 * N],
         "anchor_query": np.stack([q[0] for q in Q]), "rel_query": np.stack([q[1] for q in Q])}
    out = {"world": w, "S": S}
    if answers:
        for name, with_chunk in (("no_write_logits", False), ("in_context_logits", True)):
            texts = [answer_prompt(w, i, with_chunk) for i in range(N)]
            _, Z = g1.run_texts(model, encode_offsets, pad_id, device, texts, [[len(t)] for t in texts],
                                LAYERS[:1], batch, want_logits=True)
            out[name] = Z
    return out


# ------------------------------------------------------------------------------------------- keys

class PartMap:
    """Per-type dev mean, PCA to d, whitened, at one layer."""

    def __init__(self, worlds, layer, d, types):
        self.li = LAYERS.index(layer)
        self.mu = {t: np.mean(np.concatenate([c["S"][t][:, self.li] for c in worlds]), axis=0) for t in types}
        X = np.concatenate([c["S"][t][:, self.li] - self.mu[t] for c in worlds for t in types])
        _, s, Vt = np.linalg.svd(X, full_matrices=False)
        d = min(d, Vt.shape[0])
        self.P = Vt[:d] / (s[:d, None] / np.sqrt(len(X)) + 1e-8)

    def z(self, H, t):
        return (H[:, self.li] - self.mu[t]) @ self.P.T


class KeyBuilder:
    def __init__(self, worlds, compose, layer, d, expand):
        self.compose, self.expand = compose, expand
        self.amap = PartMap(worlds, layer, d, ("anchor_write", "anchor_query"))
        self.rmap = PartMap(worlds, layer, d, ("rel_write", "rel_query"))
        self.nmap = PartMap(worlds, layer, d, ("name_bind", "name_fact"))
        da, dr = self.amap.P.shape[0], self.rmap.P.shape[0]
        self.q = max(1, min(da, dr) // 4)
        dim = {"concat": da + dr, "outer": self.q ** 2, "phrase": dr, "anchor": da}[compose]
        self.R1 = np.random.default_rng(7).normal(size=(EXPAND_N, dim))
        self.Rn = np.random.default_rng(8).normal(size=(EXPAND_N, self.nmap.P.shape[0]))

    def _sharp(self, z, R):
        return g1.topk_sparse(z @ R.T, TOPK) if self.expand else g1._unit(z)

    def hop1(self, S, side):
        a, r = self.amap.z(S["anchor_" + side], "anchor_" + side), self.rmap.z(S["rel_" + side], "rel_" + side)
        if self.compose == "concat":
            z = np.concatenate([a, r], axis=1)
        elif self.compose == "outer":
            q = self.q
            z = np.einsum("ni,nj->nij", a[:, :q], r[:, :q]).reshape(len(a), -1)
        elif self.compose == "phrase":
            z = r
        else:
            z = a
        return self._sharp(z, self.R1)

    def name(self, S, t):
        return self._sharp(self.nmap.z(S[t], t), self.Rn)

    def clean(self, v):
        return g1.topk_sparse(v, TOPK) if self.expand else g1._unit(v)


# ------------------------------------------------------------------------------------------- memory

def derangement(seed):
    rng = np.random.default_rng(seed)
    p = np.arange(N)
    while np.any(p == np.arange(N)):
        p = rng.permutation(N)
    return p


def memories(c, kb: KeyBuilder, shuffle_binding=False, city_mu=None, value_layer=None):
    """Write M1 (hop 1), M2 (name -> city one-hot) and optionally M3 (name -> centred city state)."""
    w, S = c["world"], c["S"]
    k1w, kbind, kf = kb.hop1(S, "write"), kb.name(S, "name_bind"), kb.name(S, "name_fact")
    perm = derangement(20_000 + w.seed) if shuffle_binding else np.arange(N)
    M1, M2 = g1.Delta(k1w.shape[1], kbind.shape[1]), g1.Delta(kf.shape[1], len(g1.CITIES))
    M3 = None
    if value_layer is not None:
        vl = LAYERS.index(value_layer)
        M3 = g1.Delta(kf.shape[1], S["city_fact"].shape[2])
    for kind, i in w.order:
        if kind == "bind":
            M1.write(k1w[i], kbind[perm[i]])
        else:
            M2.write(kf[i], g1.city_onehot(w, i))
            if M3 is not None:
                M3.write(kf[i], S["city_fact"][i, vl] - city_mu[value_layer])
    return M1, M2, M3, kf


def eval_1b(c, kb, shuffle_binding=False):
    w = c["world"]
    M1, M2, _, kf = memories(c, kb, shuffle_binding)
    kq = kb.hop1(c["S"], "query")
    cid = np.array([g1.CITIES.index(x) for x in w.cities])
    two = hop1 = 0
    for i in range(N):
        v = kb.clean(M1.read(kq[i]))
        two += int(np.argmax(M2.read(v)[cid]) == i)
        hop1 += int(np.argmax(kf @ v) == i)
    one = sum(int(np.argmax(M2.read(kf[i])[cid]) == i) for i in range(N))  # key of the stored fact itself
    return {"two_hop": two, "hop1": hop1, "one_hop_stored_key": one, "n": N}


def write_vectors(c, kb, city_mu, value_layer, kind):
    """Unit vectors to write into the stream for each question of world c."""
    S = c["S"]
    vl = LAYERS.index(value_layer)
    if kind == "memory":
        M1, _, M3, _ = memories(c, kb, city_mu=city_mu, value_layer=value_layer)
        kq = kb.hop1(S, "query")
        return g1._unit(np.stack([M3.read(kb.clean(M1.read(kq[i]))) for i in range(N)]))
    true = S["city_fact"][:, vl] - city_mu[value_layer]
    if kind == "oracle":
        return g1._unit(true)
    if kind == "wrong":
        return g1._unit(true[derangement(30_000 + c["world"].seed)])
    raise ValueError(kind)


def score_logits(Z, w, first_id):
    cand = np.array([first_id(x) for x in w.cities])
    way8 = int((Z[:, cand].argmax(axis=1) == np.arange(N)).sum())
    top1 = int((Z.argmax(axis=1) == cand).sum())
    return way8, top1


def acc(rows, k):
    return sum(r[k] for r in rows) / sum(r["n"] for r in rows)


# ------------------------------------------------------------------------------------------- run

def run(model, encode_offsets, first_id, pad_id, device, fit_seeds=FIT_SEEDS, select_seeds=SELECT_SEEDS,
        test_seeds=TEST_SEEDS, grid1b=GRID_1B, grid2=GRID_2, batch=8, log=print):
    t0 = time.time()
    get = lambda seeds, names, ans: [collect_world(model, encode_offsets, pad_id, device, make_world(s, names),
                                                   batch, ans) for s in seeds]
    fit, sel = get(fit_seeds, g1.DEV_NAMES, False), get(select_seeds, g1.DEV_NAMES, False)
    test = get(test_seeds, g1.TEST_NAMES, True)
    log(f"states collected ({time.time() - t0:.0f}s)")
    dev = fit + sel
    n = len(test) * N

    # ---- Gate 1b
    sel1 = [{"compose": cp, "layer": L, "d": d, "expand": ex,
             "select_two_hop": acc([eval_1b(c, KeyBuilder(fit, cp, L, d, ex)) for c in sel], "two_hop")}
            for cp, L, d, ex in grid1b]
    b = max(sel1, key=lambda r: r["select_two_hop"])
    kb = KeyBuilder(dev, b["compose"], b["layer"], b["d"], b["expand"])
    main = [eval_1b(c, kb) for c in test]
    shuf = [eval_1b(c, kb, shuffle_binding=True) for c in test]
    abl = {cp: [eval_1b(c, KeyBuilder(dev, cp, b["layer"], b["d"], b["expand"])) for c in test]
           for cp in ("phrase", "anchor")}
    no_write = sum(score_logits(c["no_write_logits"], c["world"], first_id)[0] for c in test) / n
    in_ctx = sum(score_logits(c["in_context_logits"], c["world"], first_id)[0] for c in test) / n
    two = acc(main, "two_hop")
    g1b = {"chosen": b, "selection": sel1, "two_hop": two,
           "two_hop_wilson95": list(g0.wilson(sum(r["two_hop"] for r in main), n)),
           "hop1": acc(main, "hop1"), "one_hop_stored_key": acc(main, "one_hop_stored_key"),
           "no_memory": no_write, "shuffled_binding": acc(shuf, "two_hop"), "in_context_ceiling": in_ctx,
           "phrase_only_two_hop": acc(abl["phrase"], "two_hop"), "phrase_only_hop1": acc(abl["phrase"], "hop1"),
           "anchor_only_two_hop": acc(abl["anchor"], "two_hop"), "anchor_only_hop1": acc(abl["anchor"], "hop1")}
    g1b["decision"] = "PASS" if (two >= PASS_ACC and no_write <= CONTROL_MAX and
                                 g1b["shuffled_binding"] <= CONTROL_MAX) else "FAIL"
    log(f"gate 1b done ({time.time() - t0:.0f}s): two-hop {two:.3f} {g1b['decision']}")

    # ---- Gate 2
    kb_fit = KeyBuilder(fit, b["compose"], b["layer"], b["d"], b["expand"])
    mu_fit = {L: np.mean(np.concatenate([c["S"]["city_fact"][:, LAYERS.index(L)] for c in fit]), 0) for L in LAYERS}
    mu_dev = {L: np.mean(np.concatenate([c["S"]["city_fact"][:, LAYERS.index(L)] for c in dev]), 0) for L in LAYERS}

    def speak(worlds, kbx, mu, Lv, alpha, kind):
        w8 = t1 = 0
        for c in worlds:
            vecs = None if kind == "none" else write_vectors(c, kbx, mu, Lv, kind)
            texts = [answer_prompt(c["world"], i) for i in range(N)]
            Z = inject_logits(model, encode_offsets, pad_id, device, texts, vecs, Lv, alpha, batch)
            a8, a1 = score_logits(Z, c["world"], first_id)
            w8, t1 = w8 + a8, t1 + a1
        m = len(worlds) * N
        return w8 / m, t1 / m

    sel2 = [{"value_layer": Lv, "alpha": al, "select_way8": speak(sel, kb_fit, mu_fit, Lv, al, "memory")[0]}
            for Lv, al in grid2]
    s2 = max(sel2, key=lambda r: r["select_way8"])
    Lv, al = s2["value_layer"], s2["alpha"]
    res2 = {kind: speak(test, kb, mu_dev, Lv, al, kind) for kind in ("memory", "oracle", "wrong")}
    g2 = {"chosen": s2, "selection": sel2,
          "memory_write_way8": res2["memory"][0], "memory_write_top1": res2["memory"][1],
          "oracle_write_way8": res2["oracle"][0], "oracle_write_top1": res2["oracle"][1],
          "wrong_write_right_city_way8": res2["wrong"][0], "wrong_write_right_city_top1": res2["wrong"][1],
          "no_write_way8": no_write, "in_context_ceiling": in_ctx,
          "required_memory_write": SPEAK_RATIO * two}
    ok = (g2["oracle_write_way8"] >= ORACLE_MIN and g2["memory_write_way8"] >= SPEAK_RATIO * two and
          g2["wrong_write_right_city_way8"] <= CONTROL_MAX and no_write <= CONTROL_MAX)
    g2["decision"] = "PASS" if ok else "FAIL"
    return {"gate1b": g1b, "gate2": g2, "n_questions": n, "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", default="results/gate2_qwen2.5-0.5b.json")
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
    res = run(model, encode_offsets, first_id, pad_id, device, batch=args.batch)
    res["model"] = args.model
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    a, b = res["gate1b"], res["gate2"]
    c = a["chosen"]
    print(f"\n{args.model}   ({res['n_questions']} test questions, chance 0.125)")
    print(f"\nGATE 1b  chosen on dev: {c['compose']}, layer {c['layer']}, d {c['d']}, expand {c['expand']}")
    print(f"  two-hop from memory {a['two_hop']:.3f}  95% [{a['two_hop_wilson95'][0]:.3f}, {a['two_hop_wilson95'][1]:.3f}]"
          f"   hop 1 {a['hop1']:.3f}")
    print(f"  controls: no memory {a['no_memory']:.3f}   shuffled binding {a['shuffled_binding']:.3f}")
    print(f"  DECISION: {a['decision']}")
    print(f"  reported: phrase-only key two-hop {a['phrase_only_two_hop']:.3f} (hop 1 {a['phrase_only_hop1']:.3f})   "
          f"anchor-only {a['anchor_only_two_hop']:.3f} (hop 1 {a['anchor_only_hop1']:.3f})   "
          f"in-context ceiling {a['in_context_ceiling']:.3f}")
    print(f"\nGATE 2  chosen on dev: write at layer {b['chosen']['value_layer']}, alpha {b['chosen']['alpha']}")
    print(f"  the model answers itself, context empty:   8-way / full-vocab top-1")
    print(f"    memory write        {b['memory_write_way8']:.3f} / {b['memory_write_top1']:.3f}"
          f"   (needs >= {b['required_memory_write']:.3f})")
    print(f"    oracle write        {b['oracle_write_way8']:.3f} / {b['oracle_write_top1']:.3f}   (needs >= 0.80)")
    print(f"    wrong-city write    {b['wrong_write_right_city_way8']:.3f} / {b['wrong_write_right_city_top1']:.3f}"
          f"   (right-city rate, needs <= 0.25)")
    print(f"    no write            {b['no_write_way8']:.3f}   (needs <= 0.25)")
    print(f"  DECISION: {b['decision']}")
    print("\n  dev selection, gate 1b (top 5):")
    for r in sorted(a["selection"], key=lambda r: -r["select_two_hop"])[:5]:
        print(f"    {r['compose']:6s} layer {r['layer']} d {r['d']:2d} expand {str(r['expand']):5s} {r['select_two_hop']:.3f}")
    print("  dev selection, gate 2:")
    for r in sorted(b["selection"], key=lambda r: -r["select_way8"]):
        print(f"    layer {r['value_layer']:2d} alpha {r['alpha']:3.1f}  {r['select_way8']:.3f}")


if __name__ == "__main__":
    main()
