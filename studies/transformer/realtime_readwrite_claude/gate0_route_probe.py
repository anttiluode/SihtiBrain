"""Gate 0 (PREREG.md): does a frozen LLM reach the same address by two routes?

For each synthetic world of 8 bindings "A's r is B.", read the hidden state at the last token of
  direct route      "... Now consider B"
  relational route  "... Now consider A's r"
at every layer, and ask whether the relational state retrieves B among the 8 targets.

Run (GPU recommended; CPU works, slowly):
    pip install torch transformers numpy
    python gate0_route_probe.py --model Qwen/Qwen2.5-0.5B --out results/gate0_qwen2.5-0.5b.json
Reported (not gated) models:
    python gate0_route_probe.py --model Qwen/Qwen2.5-0.5B-Instruct --out results/gate0_qwen2.5-0.5b-instruct.json
    python gate0_route_probe.py --model Qwen/Qwen2.5-1.5B --out results/gate0_qwen2.5-1.5b.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

NAMES = [
    "Anna", "Ben", "Clara", "David", "Emma", "Frank", "Grace", "Henry", "Irene", "Jack",
    "Kate", "Leo", "Maria", "Nick", "Olivia", "Paul", "Rachel", "Sam", "Tina", "Victor",
    "Wendy", "Adam", "Bella", "Carl", "Diana", "Eric", "Fiona", "George", "Hannah", "Ian",
    "Julia", "Kevin", "Laura", "Mark", "Nina", "Oscar", "Peter", "Rose", "Steve", "Tom",
    "Vera", "Walter", "Alice", "Bob", "Chris", "Daniel", "Ella", "Felix", "Gina", "Hugo",
    "Iris", "James", "Karen", "Lucas", "Mia", "Noah", "Owen", "Paula", "Ryan", "Sara",
    "Tim", "Uma", "Vince", "Will", "Zoe", "Alan", "Beth", "Colin", "Dora", "Ethan",
    "Faye", "Gary", "Helen", "Ivan", "Jane", "Kurt", "Lily", "Max", "Nora", "Otto",
    "Pia", "Ralph", "Sophie", "Ted", "Ursula", "Vicky", "Wade", "Amy", "Bruce", "Cindy",
    "Dean", "Eva", "Fred", "Gail", "Hank", "Ida", "Jon", "Kim", "Liam", "Molly",
]
RELATIONS = ["brother", "sister", "friend", "boss", "neighbour", "teacher", "cousin", "partner"]
N_BIND = 8
DEV_SEEDS = list(range(0, 50))
TEST_SEEDS = list(range(1000, 1150))
PASS_ACC = 0.80
CONTROL_MAX = 0.25


# --------------------------------------------------------------------------------------- worlds

@dataclass
class World:
    seed: int
    anchors: list[str]
    relations: list[str]
    targets: list[str]
    context: str
    nobind_context: str
    prompts: dict[str, list[str]] = field(default_factory=dict)


def make_world(seed: int, n: int = N_BIND) -> World:
    rng = random.Random(seed)
    names = rng.sample(NAMES, 2 * n)
    anchors, targets = names[:n], names[n:]
    relations = [rng.choice(RELATIONS) for _ in range(n)]
    sentences = [f"{a}'s {r} is {b}." for a, r, b in zip(anchors, relations, targets)]
    order = list(range(n))
    rng.shuffle(order)
    context = " ".join(sentences[i] for i in order)
    listed = names[:]
    rng.shuffle(listed)
    nobind_context = "People: " + ", ".join(listed) + "."
    w = World(seed, anchors, relations, targets, context, nobind_context)
    p = lambda ctx, ref: f"{ctx}\nNow consider {ref}"
    w.prompts = {
        "direct_target": [p(context, b) for b in targets],
        "direct_anchor": [p(context, a) for a in anchors],
        "relational": [p(context, f"{a}'s {r}") for a, r in zip(anchors, relations)],
        "nobind_direct_target": [p(nobind_context, b) for b in targets],
        "nobind_relational": [p(nobind_context, f"{a}'s {r}") for a, r in zip(anchors, relations)],
    }
    return w


# --------------------------------------------------------------------------------------- metrics

def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def retrieval(D: np.ndarray, R: np.ndarray, centre: bool = True) -> dict:
    """D: [n,d] candidate states (row i is the correct answer for query i). R: [n,d] queries."""
    if centre:
        mu = D.mean(axis=0, keepdims=True)
        D, R = D - mu, R - mu
    S = _unit(R) @ _unit(D).T  # [n_query, n_cand]
    n = S.shape[0]
    correct = S.argmax(axis=1) == np.arange(n)
    wrong = S.copy()
    wrong[np.arange(n), np.arange(n)] = -np.inf
    margin = S[np.arange(n), np.arange(n)] - wrong.max(axis=1)
    return {"n_correct": int(correct.sum()), "n": n, "margin": float(margin.mean())}


def retrieval16(T: np.ndarray, A: np.ndarray, R: np.ndarray) -> dict:
    """16-way: candidates are 8 targets then 8 anchors, centred by the mean of all 16."""
    C = np.concatenate([T, A], axis=0)
    mu = C.mean(axis=0, keepdims=True)
    S = _unit(R - mu) @ _unit(C - mu).T
    n = R.shape[0]
    pick = S.argmax(axis=1)
    return {"n_target": int((pick == np.arange(n)).sum()),
            "n_own_anchor": int((pick == np.arange(n) + n).sum()), "n": n}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


# --------------------------------------------------------------------------------------- model

def last_token_states(model, encode, texts: list[str], pad_id: int, device, batch: int = 16) -> np.ndarray:
    """Hidden states at the last token of each text, all layers. Returns [n, L+1, d] float32.

    Left-pads within a batch and passes explicit position_ids so padded rows see the same
    positions as an unpadded run (tested in tests/test_gate0.py).
    """
    import torch

    out = []
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
        hs = torch.stack([h[:, -1, :] for h in o.hidden_states], dim=1)  # [b, L+1, d]
        out.append(hs.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def score_world(model, encode, pad_id, device, w: World, batch: int) -> dict:
    keys = list(w.prompts)
    texts = [t for k in keys for t in w.prompts[k]]
    H = last_token_states(model, encode, texts, pad_id, device, batch)
    st = {k: H[i * N_BIND:(i + 1) * N_BIND] for i, k in enumerate(keys)}
    n_layers = H.shape[1]
    per_layer = []
    for l in range(n_layers):
        T, A, R = st["direct_target"][:, l], st["direct_anchor"][:, l], st["relational"][:, l]
        per_layer.append({
            "centred": retrieval(T, R, centre=True),
            "raw": retrieval(T, R, centre=False),
            "way16": retrieval16(T, A, R),
            "nobind": retrieval(st["nobind_direct_target"][:, l], st["nobind_relational"][:, l], centre=True),
        })
    return {"seed": w.seed, "layers": per_layer}


def aggregate(worlds: list[dict]) -> list[dict]:
    n_layers = len(worlds[0]["layers"])
    agg = []
    for l in range(n_layers):
        rows = [w["layers"][l] for w in worlds]
        tot = sum(r["centred"]["n"] for r in rows)
        agg.append({
            "layer": l,
            "acc_centred": sum(r["centred"]["n_correct"] for r in rows) / tot,
            "acc_raw": sum(r["raw"]["n_correct"] for r in rows) / tot,
            "margin_centred": float(np.mean([r["centred"]["margin"] for r in rows])),
            "way16_target": sum(r["way16"]["n_target"] for r in rows) / tot,
            "way16_own_anchor": sum(r["way16"]["n_own_anchor"] for r in rows) / tot,
            "acc_nobind": sum(r["nobind"]["n_correct"] for r in rows) / tot,
            "n_correct": sum(r["centred"]["n_correct"] for r in rows),
            "n": tot,
        })
    return agg


def run(model, encode, pad_id, device, dev_seeds, test_seeds, batch=16, log=print) -> dict:
    t0 = time.time()
    dev = [score_world(model, encode, pad_id, device, make_world(s), batch) for s in dev_seeds]
    log(f"dev done ({len(dev)} worlds, {time.time() - t0:.0f}s)")
    test = [score_world(model, encode, pad_id, device, make_world(s), batch) for s in test_seeds]
    log(f"test done ({len(test)} worlds, {time.time() - t0:.0f}s)")
    dev_agg, test_agg = aggregate(dev), aggregate(test)
    l_star = int(np.argmax([a["acc_centred"] for a in dev_agg]))
    t = test_agg[l_star]
    lo, hi = wilson(t["n_correct"], t["n"])
    passed = t["acc_centred"] >= PASS_ACC and t["acc_nobind"] <= CONTROL_MAX
    return {
        "l_star_from_dev": l_star,
        "test_at_l_star": {**t, "wilson95": [lo, hi]},
        "decision": "PASS" if passed else "FAIL",
        "criteria": {"acc_centred_min": PASS_ACC, "nobind_max": CONTROL_MAX, "chance": 1 / N_BIND},
        "dev_by_layer": dev_agg,
        "test_by_layer": test_agg,
        "seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", default="results/gate0_qwen2.5-0.5b.json")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    args = ap.parse_args()

    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(device).eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    encode = lambda s: tok(s, add_special_tokens=True)["input_ids"]

    res = run(model, encode, pad_id, device, DEV_SEEDS, TEST_SEEDS, args.batch)
    res.update({"model": args.model, "dtype": args.dtype, "device": device,
                "example_world": make_world(TEST_SEEDS[0]).prompts["relational"][0]})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    t = res["test_at_l_star"]
    print(f"\n{args.model}: layer l* = {res['l_star_from_dev']} (chosen on dev)")
    print(f"  test centred accuracy {t['acc_centred']:.3f}  95% [{t['wilson95'][0]:.3f}, {t['wilson95'][1]:.3f}]"
          f"   no-binding control {t['acc_nobind']:.3f}   chance {1 / N_BIND:.3f}")
    print(f"  16-way: target {t['way16_target']:.3f}, own anchor {t['way16_own_anchor']:.3f}")
    print(f"  DECISION: {res['decision']}")
    print("\n layer  dev_acc  test_acc  test_raw  nobind  16:tgt  16:anchor")
    for d, a in zip(res["dev_by_layer"], res["test_by_layer"]):
        print(f"  {a['layer']:3d}   {d['acc_centred']:.3f}    {a['acc_centred']:.3f}     {a['acc_raw']:.3f}    "
              f"{a['acc_nobind']:.3f}   {a['way16_target']:.3f}   {a['way16_own_anchor']:.3f}")


if __name__ == "__main__":
    main()
