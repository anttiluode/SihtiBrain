"""Gate 0b — POST-RESULT diagnostics for the Gate 0 FAIL. Written after seeing the result; it cannot
change the Gate 0 decision. It separates three readings of the 41% (RESULT_GATE0.md):

  A. behaviour  — does the model itself resolve the two-hop?  "...Now consider Anna's brother. Anna's brother is" -> Ben?
                  scored as (i) full-vocabulary top-1 and (ii) 8-way choice among the world's target names.
  B. position   — is the referent present at the position where it must be produced (the "is" token)?
                  same centred retrieval as Gate 0, query state read there instead of at "brother".
  C. subspace   — is there a linear map, fitted on dev worlds only, under which the relational state retrieves
                  the right target in held-out test worlds?  Ridge regression, lambda chosen on a dev split.
                  Control: the same pipeline on no-binding contexts must stay at chance.

Run:
    python gate0b_diagnostics.py --model Qwen/Qwen2.5-0.5B --out results/gate0b_qwen2.5-0.5b.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import gate0_route_probe as g

FIT_SEEDS, VAL_SEEDS = g.DEV_SEEDS[:35], g.DEV_SEEDS[35:]
LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0]  # times mean squared row norm


def resolve_prompts(w: g.World, nobind: bool = False) -> list[str]:
    ctx = w.nobind_context if nobind else w.context
    return [f"{ctx}\nNow consider {a}'s {r}. {a}'s {r} is" for a, r in zip(w.anchors, w.relations)]


def states_and_logits(model, encode, texts, pad_id, device, batch=16):
    """Like g.last_token_states, plus the last-position logits. Returns ([n,L+1,d], [n,V])."""
    import torch

    H, Z = [], []
    for s in range(0, len(texts), batch):
        ids = [encode(t) for t in texts[s:s + batch]]
        L = max(len(x) for x in ids)
        inp = torch.full((len(ids), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(ids), L), dtype=torch.long)
        for i, x in enumerate(ids):
            inp[i, L - len(x):] = torch.tensor(x)
            mask[i, L - len(x):] = 1
        pos = (mask.cumsum(dim=1) - 1).clamp(min=0)
        with torch.no_grad():
            o = model(input_ids=inp.to(device), attention_mask=mask.to(device),
                      position_ids=pos.to(device), output_hidden_states=True)
        H.append(torch.stack([h[:, -1, :] for h in o.hidden_states], 1).float().cpu().numpy())
        Z.append(o.logits[:, -1, :].float().cpu().numpy())
    return np.concatenate(H), np.concatenate(Z)


def collect(model, encode, first_id, pad_id, device, seeds, batch):
    """Per world: direct-target, relational, resolve states; nobind versions; behaviour scores."""
    out = []
    for s in seeds:
        w = g.make_world(s)
        texts = (w.prompts["direct_target"] + w.prompts["relational"] + resolve_prompts(w)
                 + w.prompts["nobind_direct_target"] + w.prompts["nobind_relational"] + resolve_prompts(w, True))
        H, Z = states_and_logits(model, encode, texts, pad_id, device, batch)
        n = g.N_BIND
        cand = np.array([first_id(b) for b in w.targets])
        Zr = Z[2 * n:3 * n]
        top1 = Zr.argmax(axis=1) == cand
        way8 = Zr[:, cand].argmax(axis=1) == np.arange(n)
        Zc = Z[5 * n:6 * n]
        out.append({
            "T": H[0:n], "R": H[n:2 * n], "Q": H[2 * n:3 * n],
            "nT": H[3 * n:4 * n], "nR": H[4 * n:5 * n], "nQ": H[5 * n:6 * n],
            "top1": int(top1.sum()), "way8": int(way8.sum()),
            "nobind_way8": int((Zc[:, cand].argmax(axis=1) == np.arange(n)).sum()),
        })
    return out


def centred(world, qkey, tkey, l):
    """Each side centred by its own world mean (a linear map then sees only within-world differences)."""
    T, Q = world[tkey][:, l], world[qkey][:, l]
    return T - T.mean(axis=0, keepdims=True), Q - Q.mean(axis=0, keepdims=True)


def acc_cosine(worlds, qkey, tkey, l):
    k = n = 0
    for w in worlds:
        r = g.retrieval(w[tkey][:, l], w[qkey][:, l], centre=True)
        k, n = k + r["n_correct"], n + r["n"]
    return k / n


def fit_ridge(worlds, qkey, tkey, l, lam):
    X = np.concatenate([centred(w, qkey, tkey, l)[1] for w in worlds])
    Y = np.concatenate([centred(w, qkey, tkey, l)[0] for w in worlds])
    # Floor the scale: at layer 0 queries ending on the same token centre to exactly zero.
    scale = max(float((X ** 2).sum(axis=1).mean()), 1e-12)
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + (lam * scale + 1e-8) * np.eye(d), X.T @ Y)


def acc_mapped(worlds, qkey, tkey, l, W):
    k = n = 0
    for w in worlds:
        Tc, Qc = centred(w, qkey, tkey, l)
        r = g.retrieval(Tc, Qc @ W, centre=False)
        k, n = k + r["n_correct"], n + r["n"]
    return k / n


def run(model, encode, first_id, pad_id, device, fit_seeds=FIT_SEEDS, val_seeds=VAL_SEEDS,
        test_seeds=g.TEST_SEEDS, batch=16, log=print):
    t0 = time.time()
    fit = collect(model, encode, first_id, pad_id, device, fit_seeds, batch)
    val = collect(model, encode, first_id, pad_id, device, val_seeds, batch)
    test = collect(model, encode, first_id, pad_id, device, test_seeds, batch)
    log(f"states collected ({time.time() - t0:.0f}s)")
    n_test = len(test) * g.N_BIND
    res = {"post_result": True,
           "A_behaviour": {"top1_vocab": sum(w["top1"] for w in test) / n_test,
                           "way8": sum(w["way8"] for w in test) / n_test,
                           "nobind_way8": sum(w["nobind_way8"] for w in test) / n_test,
                           "chance_way8": 1 / g.N_BIND},
           "by_layer": []}
    n_layers = test[0]["T"].shape[1]
    for l in range(n_layers):
        row = {"layer": l,
               "B_cos_at_brother": acc_cosine(test, "R", "T", l),
               "B_cos_at_is": acc_cosine(test, "Q", "T", l)}
        for qkey, name in (("R", "brother"), ("Q", "is")):
            best = max(LAMBDAS, key=lambda lam: acc_mapped(val, qkey, "T", l, fit_ridge(fit, qkey, "T", l, lam)))
            W = fit_ridge(fit + val, qkey, "T", l, best)
            row[f"C_ridge_at_{name}"] = acc_mapped(test, qkey, "T", l, W)
            row[f"C_lambda_at_{name}"] = best
            Wn = fit_ridge(fit + val, "n" + qkey, "nT", l, best)
            row[f"C_nobind_at_{name}"] = acc_mapped(test, "n" + qkey, "nT", l, Wn)
        res["by_layer"].append(row)
    res["seconds"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", default="results/gate0b_qwen2.5-0.5b.json")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    encode = lambda s: tok(s, add_special_tokens=True)["input_ids"]
    first_id = lambda name: tok(" " + name, add_special_tokens=False)["input_ids"][0]
    firsts = [first_id(n) for n in g.NAMES]
    if len(set(firsts)) < len(firsts):
        print(f"note: {len(firsts) - len(set(firsts))} name pairs share a first token; way8 ties are possible")

    res = run(model, encode, first_id, pad_id, device, batch=args.batch)
    res["model"] = args.model
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    a = res["A_behaviour"]
    print(f"\n{args.model} — POST-RESULT diagnostics (cannot change Gate 0)")
    print(f"A. behaviour: '...A's r is' -> B   top-1 vocab {a['top1_vocab']:.3f}   8-way {a['way8']:.3f}"
          f"   (no-binding 8-way {a['nobind_way8']:.3f}, chance {a['chance_way8']:.3f})")
    print("\n layer  B:cos@brother  B:cos@is   C:ridge@brother  C:ridge@is   C:nobind@brother  C:nobind@is")
    for r in res["by_layer"]:
        print(f"  {r['layer']:3d}      {r['B_cos_at_brother']:.3f}       {r['B_cos_at_is']:.3f}"
              f"        {r['C_ridge_at_brother']:.3f}           {r['C_ridge_at_is']:.3f}"
              f"          {r['C_nobind_at_brother']:.3f}           {r['C_nobind_at_is']:.3f}")


if __name__ == "__main__":
    main()
