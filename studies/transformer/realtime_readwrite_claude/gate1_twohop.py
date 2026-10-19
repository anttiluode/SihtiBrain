"""Gate 1 (PREREG_GATE1.md): two-hop recall from a live memory after the context is gone.

Frozen Qwen reads one chunk of bindings ("Anna's brother is Ben.") and facts ("Ben lives in Oslo."); two
delta memories are written from its states; the chunk is discarded; then "Where does Anna's brother live?"
is answered through the memories alone.

    python gate1_twohop.py --model Qwen/Qwen2.5-0.5B --out results/gate1_qwen2.5-0.5b.json
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

DEV_NAMES, TEST_NAMES = g0.NAMES[:60], g0.NAMES[60:]
CITIES = ["Oslo", "Rome", "Paris", "Berlin", "Madrid", "Vienna", "Prague", "Dublin", "Lisbon", "Athens",
          "Warsaw", "Helsinki", "Stockholm", "Copenhagen", "Brussels", "Amsterdam", "Zurich", "Budapest",
          "Tokyo", "Seoul", "Beijing", "Delhi", "Cairo", "Lagos", "Nairobi", "Lima", "Bogota", "Santiago",
          "Toronto", "Chicago", "Boston", "Denver", "Sydney", "Perth", "Auckland", "Dubai", "Istanbul",
          "Moscow", "Riga", "Tallinn"]
N = 8
FIT_SEEDS, SELECT_SEEDS = list(range(0, 30)), list(range(30, 50))
TEST_SEEDS = list(range(1000, 1100))
LAYERS = (14, 17, 18)
GRID = list(itertools.product(LAYERS, (64, 128, 256), (True, False)))
EXPAND_N, TOPK = 2048, 32
PASS_ACC, CONTROL_MAX = 0.80, 0.25
STATE_TYPES = ("rel_write", "name_bind", "name_fact", "rel_query", "name_query", "fact_end", "qrel_end", "qdir_end")
MAIN_TYPES = ("rel_write", "name_bind", "name_fact", "rel_query", "name_query")
SE_TYPES = ("fact_end", "qrel_end", "qdir_end")


# ------------------------------------------------------------------------------------------- worlds

@dataclass
class World:
    seed: int
    anchors: list
    relations: list
    targets: list
    cities: list           # cities[i] is where targets[i] lives
    chunk: str
    rel_end: list          # char end of "A's r" in binding i
    bind_name_end: list    # char end of B in binding i
    fact_name_end: list    # char end of B in fact i
    fact_dot: list         # char index just past the "." of fact i
    order: list            # reading order: ("bind"|"fact", i)


def make_world(seed: int, names: list) -> World:
    rng = random.Random(seed)
    people = rng.sample(names, 2 * N)
    anchors, targets = people[:N], people[N:]
    relations = [rng.choice(g0.RELATIONS) for _ in range(N)]
    cities = rng.sample(CITIES, N)
    order = [("bind", i) for i in range(N)] + [("fact", i) for i in range(N)]
    rng.shuffle(order)
    text, rel_end, bind_end, fact_end, fact_dot = "", [0] * N, [0] * N, [0] * N, [0] * N
    for kind, i in order:
        if text:
            text += " "
        if kind == "bind":
            ref = f"{anchors[i]}'s {relations[i]}"
            rel_end[i] = len(text) + len(ref)
            text += f"{ref} is "
            text += targets[i]
            bind_end[i] = len(text)
            text += "."
        else:
            text += targets[i]
            fact_end[i] = len(text)
            text += f" lives in {cities[i]}."
            fact_dot[i] = len(text)
    return World(seed, anchors, relations, targets, cities, text, rel_end, bind_end, fact_end, fact_dot, order)


def q_rel(w, i):
    ref = f"{w.anchors[i]}'s {w.relations[i]}"
    return f"Where does {ref} live?", len("Where does ") + len(ref)


def q_dir(w, i):
    return f"Where does {w.targets[i]} live?", len("Where does ") + len(w.targets[i])


def answer_prompt(w, i, with_chunk: bool):
    ref = f"{w.anchors[i]}'s {w.relations[i]}"
    p = f"Q: Where does {ref} live?\nA: {ref} lives in"
    return (w.chunk + "\n" + p) if with_chunk else p


# ------------------------------------------------------------------------------------------- model I/O

def token_at(offsets, char_end: int) -> int:
    """Index of the token that contains character char_end-1."""
    for t, (s, e) in enumerate(offsets):
        if s <= char_end - 1 < e:
            return t
    raise ValueError(f"no token covers char {char_end - 1}")


def run_texts(model, encode_offsets, pad_id, device, texts, char_ends, layers, batch=8, want_logits=False):
    """For each text, hidden states at the tokens ending at the given chars, for the given layers.
    Returns (list of [n_pos, n_layers, d] arrays, optional [n_texts, V] last-position logits)."""
    import torch

    states, logits = [], []
    for s in range(0, len(texts), batch):
        enc = [encode_offsets(t) for t in texts[s:s + batch]]
        L = max(len(ids) for ids, _ in enc)
        inp = torch.full((len(enc), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(enc), L), dtype=torch.long)
        for b, (ids, _) in enumerate(enc):
            inp[b, L - len(ids):] = torch.tensor(ids)
            mask[b, L - len(ids):] = 1
        pos = (mask.cumsum(dim=1) - 1).clamp(min=0)
        with torch.no_grad():
            o = model(input_ids=inp.to(device), attention_mask=mask.to(device), position_ids=pos.to(device),
                      output_hidden_states=True)
        for b, (ids, offs) in enumerate(enc):
            shift = L - len(ids)
            idx = [shift + token_at(offs, c) for c in char_ends[s + b]]
            states.append(np.stack([o.hidden_states[l][b, idx].float().cpu().numpy() for l in layers], axis=1))
            if want_logits:
                logits.append(o.logits[b, -1].float().cpu().numpy())
    return states, (np.stack(logits) if want_logits else None)


def collect_world(model, encode_offsets, first_id, pad_id, device, w: World, layers, batch, answers: bool):
    ch_pos = w.rel_end + w.bind_name_end + w.fact_name_end + w.fact_dot
    (chunk_states,), _ = run_texts(model, encode_offsets, pad_id, device, [w.chunk], [ch_pos], layers, batch)
    qr = [q_rel(w, i) for i in range(N)]
    qd = [q_dir(w, i) for i in range(N)]
    qtexts = [t for t, _ in qr] + [t for t, _ in qd]
    qpos = [[c, len(t)] for t, c in qr] + [[c, len(t)] for t, c in qd]
    qs, _ = run_texts(model, encode_offsets, pad_id, device, qtexts, qpos, layers, batch)
    S = {"rel_write": chunk_states[0:N], "name_bind": chunk_states[N:2 * N],
         "name_fact": chunk_states[2 * N:3 * N], "fact_end": chunk_states[3 * N:4 * N],
         "rel_query": np.stack([qs[i][0] for i in range(N)]), "qrel_end": np.stack([qs[i][1] for i in range(N)]),
         "name_query": np.stack([qs[N + i][0] for i in range(N)]), "qdir_end": np.stack([qs[N + i][1] for i in range(N)])}
    out = {"world": w, "S": S}
    if answers:
        cand = np.array([first_id(c) for c in w.cities])
        for name, with_chunk in (("no_memory", False), ("in_context", True)):
            texts = [answer_prompt(w, i, with_chunk) for i in range(N)]
            _, Z = run_texts(model, encode_offsets, pad_id, device, texts, [[len(t)] for t in texts], layers[:1],
                             batch, want_logits=True)
            out[name] = int((Z[:, cand].argmax(axis=1) == np.arange(N)).sum())
    return out


# ------------------------------------------------------------------------------------------- keys + memory

def _unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(n > 0, x / np.maximum(n, 1e-12), x)


def topk_sparse(x, k):
    x = np.maximum(x, 0.0)
    if x.shape[-1] > k:
        thr = np.partition(x, -k, axis=-1)[..., -k][..., None]
        x = np.where(x >= thr, x, 0.0)
    return _unit(x)


class KeyMap:
    def __init__(self, worlds, li, d, whiten, types, expand=True, seed=7):
        self.li, self.expand = li, expand
        self.mu = {t: np.mean(np.concatenate([w["S"][t][:, li] for w in worlds]), axis=0) for t in types}
        X = np.concatenate([w["S"][t][:, li] - self.mu[t] for w in worlds for t in types])
        _, s, Vt = np.linalg.svd(X, full_matrices=False)
        d = min(d, Vt.shape[0])
        self.P = Vt[:d] / (s[:d, None] / np.sqrt(len(X)) + 1e-8) if whiten else Vt[:d]
        self.R = np.random.default_rng(seed).normal(size=(EXPAND_N, d))

    def __call__(self, H, t):
        z = (H[:, self.li] - self.mu[t]) @ self.P.T
        return topk_sparse(z @ self.R.T, TOPK) if self.expand else _unit(z)

    def clean(self, v):
        return topk_sparse(v, TOPK) if self.expand else _unit(v)


class Delta:
    def __init__(self, dk, dv):
        self.M = np.zeros((dv, dk))

    def write(self, k, v, beta=1.0):
        self.M += beta * np.outer(v - self.M @ k, k)

    def read(self, k):
        return self.M @ k


def city_onehot(w, i):
    v = np.zeros(len(CITIES))
    v[CITIES.index(w.cities[i])] = 1.0
    return v


def eval_world(c, km: KeyMap, shuffle_binding=False, seed=0):
    w, S = c["world"], c["S"]
    kr, kb, kf = km(S["rel_write"], "rel_write"), km(S["name_bind"], "name_bind"), km(S["name_fact"], "name_fact")
    kq, kd = km(S["rel_query"], "rel_query"), km(S["name_query"], "name_query")
    perm = np.arange(N)
    if shuffle_binding:
        rng = np.random.default_rng(10_000 + w.seed + seed)
        while np.any(perm == np.arange(N)):
            perm = rng.permutation(N)
    D = kr.shape[1]
    M1, M2 = Delta(D, D), Delta(D, len(CITIES))
    for kind, i in w.order:  # one pass, reading order
        if kind == "bind":
            M1.write(kr[i], kb[perm[i]])
        else:
            M2.write(kf[i], city_onehot(w, i))
    cid = np.array([CITIES.index(x) for x in w.cities])
    two = hop1 = one = 0
    for i in range(N):
        v = km.clean(M1.read(kq[i]))
        two += int(np.argmax(M2.read(v)[cid]) == i)
        hop1 += int(np.argmax(kf @ v) == i)
        one += int(np.argmax(M2.read(kd[i])[cid]) == i)
    return {"two_hop": two, "hop1": hop1, "one_hop": one, "n": N}


def eval_sentence_end(c, km: KeyMap):
    w, S = c["world"], c["S"]
    kf, qr, qd = km(S["fact_end"], "fact_end"), km(S["qrel_end"], "qrel_end"), km(S["qdir_end"], "qdir_end")
    M = Delta(kf.shape[1], len(CITIES))
    for kind, i in w.order:
        if kind == "fact":
            M.write(kf[i], city_onehot(w, i))
    cid = np.array([CITIES.index(x) for x in w.cities])
    return {"two_hop": sum(int(np.argmax(M.read(qr[i])[cid]) == i) for i in range(N)),
            "one_hop": sum(int(np.argmax(M.read(qd[i])[cid]) == i) for i in range(N)), "n": N}


def oracle(c):
    w = c["world"]
    rel = {f"{a}'s {r}": b for a, r, b in zip(w.anchors, w.relations, w.targets)}
    lives = dict(zip(w.targets, w.cities))
    return sum(int(lives[rel[f"{w.anchors[i]}'s {w.relations[i]}"]] == w.cities[i]) for i in range(N))


def acc(rows, key):
    return sum(r[key] for r in rows) / sum(r["n"] for r in rows)


# ------------------------------------------------------------------------------------------- run

def run(model, encode_offsets, first_id, pad_id, device, fit_seeds=FIT_SEEDS, select_seeds=SELECT_SEEDS,
        test_seeds=TEST_SEEDS, grid=GRID, layers=LAYERS, batch=8, log=print):
    t0 = time.time()
    get = lambda seeds, names, ans: [collect_world(model, encode_offsets, first_id, pad_id, device,
                                                   make_world(s, names), layers, batch, ans) for s in seeds]
    fit, sel = get(fit_seeds, DEV_NAMES, False), get(select_seeds, DEV_NAMES, False)
    test = get(test_seeds, TEST_NAMES, True)
    log(f"states collected ({time.time() - t0:.0f}s)")

    selection = []
    for L, d, wh in grid:
        km = KeyMap(fit, layers.index(L), d, wh, MAIN_TYPES)
        selection.append({"layer": L, "d": d, "whiten": wh,
                          "select_two_hop": acc([eval_world(c, km) for c in sel], "two_hop")})
    best = max(selection, key=lambda r: r["select_two_hop"])
    L, d, wh = best["layer"], best["d"], best["whiten"]
    li = layers.index(L)
    km = KeyMap(fit + sel, li, d, wh, MAIN_TYPES)
    km_raw = KeyMap(fit + sel, li, d, wh, MAIN_TYPES, expand=False)
    km_se = KeyMap(fit + sel, li, d, wh, SE_TYPES)

    main = [eval_world(c, km) for c in test]
    shuf = [eval_world(c, km, shuffle_binding=True) for c in test]
    raw = [eval_world(c, km_raw) for c in test]
    se = [eval_sentence_end(c, km_se) for c in test]
    n = len(test) * N
    two = acc(main, "two_hop")
    nomem = sum(c["no_memory"] for c in test) / n
    shuf_acc = acc(shuf, "two_hop")
    passed = two >= PASS_ACC and nomem <= CONTROL_MAX and shuf_acc <= CONTROL_MAX
    lo, hi = g0.wilson(sum(r["two_hop"] for r in main), n)
    return {
        "chosen": {"layer": L, "d": d, "whiten": wh}, "selection": selection,
        "test": {"two_hop": two, "two_hop_wilson95": [lo, hi], "no_memory": nomem, "shuffled_binding": shuf_acc,
                 "in_context_ceiling": sum(c["in_context"] for c in test) / n,
                 "one_hop": acc(main, "one_hop"), "hop1_lands_on_right_name": acc(main, "hop1"),
                 "raw_keys_two_hop": acc(raw, "two_hop"), "raw_keys_one_hop": acc(raw, "one_hop"),
                 "sentence_end_two_hop": acc(se, "two_hop"), "sentence_end_one_hop": acc(se, "one_hop"),
                 "oracle_table": sum(oracle(c) for c in test) / n, "n_questions": n},
        "decision": "PASS" if passed else "FAIL",
        "criteria": {"two_hop_min": PASS_ACC, "controls_max": CONTROL_MAX, "chance": 1 / N},
        "seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", default="results/gate1_qwen2.5-0.5b.json")
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
    firsts = [first_id(c) for c in CITIES]
    if len(set(firsts)) < len(firsts):
        print(f"note: {len(firsts) - len(set(firsts))} city pairs share a first token; 8-way ties possible")

    res = run(model, encode_offsets, first_id, pad_id, device, batch=args.batch)
    res["model"] = args.model
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    t, ch = res["test"], res["chosen"]
    print(f"\n{args.model}: chosen on dev  layer {ch['layer']}, d {ch['d']}, whiten {ch['whiten']}")
    print(f"  TWO-HOP from memory, context gone: {t['two_hop']:.3f}  95% [{t['two_hop_wilson95'][0]:.3f}, "
          f"{t['two_hop_wilson95'][1]:.3f}]   ({t['n_questions']} questions, chance 0.125)")
    print(f"  controls: no memory {t['no_memory']:.3f}   shuffled binding {t['shuffled_binding']:.3f}")
    print(f"  DECISION: {res['decision']}")
    print(f"\n  reported: in-context ceiling {t['in_context_ceiling']:.3f}   one hop {t['one_hop']:.3f}   "
          f"hop 1 lands on right name {t['hop1_lands_on_right_name']:.3f}")
    print(f"            raw keys two-hop {t['raw_keys_two_hop']:.3f} / one-hop {t['raw_keys_one_hop']:.3f}   "
          f"sentence-end keys two-hop {t['sentence_end_two_hop']:.3f} / one-hop {t['sentence_end_one_hop']:.3f}   "
          f"oracle table {t['oracle_table']:.3f}")
    print("\n  dev selection (two-hop on select worlds):")
    for r in sorted(res["selection"], key=lambda r: -r["select_two_hop"]):
        print(f"    layer {r['layer']:2d}  d {r['d']:3d}  whiten {str(r['whiten']):5s}  {r['select_two_hop']:.3f}")


if __name__ == "__main__":
    main()
