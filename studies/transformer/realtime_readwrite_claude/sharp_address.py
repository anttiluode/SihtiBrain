"""Sharp addresses (PREREG_SHARP.md): Sol's Gate 2 world and memory, with the phase key passed
through a fixed sparse nonlinear expansion before the delta memory.

    python sharp_address.py --out results/sharp_address.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from vendor import tem_addressing as sol

SEEDS = range(1000, 1020)
PRIMARY = (512, 16)
SWEEP = [(n, k) for n in (64, 256, 1024) for k in (4, 16, 64) if k < n]


class Expansion:
    """p = unit( topk( ReLU(W g) ) ). linear=True gives unit(W g): same width, no nonlinearity."""

    def __init__(self, n: int, k: int, linear: bool = False, seed: int = 7):
        self.W = np.random.default_rng(seed).normal(size=(n, sol.DIM))
        self.n, self.k, self.linear = n, k, linear

    def __call__(self, g: np.ndarray) -> np.ndarray:
        h = self.W @ g
        if not self.linear:
            h = np.maximum(h, 0.0)
            if self.k < self.n:
                h[np.argsort(h)[:-self.k]] = 0.0
        nrm = np.linalg.norm(h)
        return h / nrm if nrm > 0 else h


class ExpandedMemory:
    """Sol's DeltaMemory writer and reader, on an expanded key."""

    def __init__(self, base_mode: str, expand: Expansion, writer: str = "delta"):
        self.base = sol.DeltaMemory(base_mode, writer)  # used only for its key()
        self.expand, self.writer = expand, writer
        self.w = np.zeros((sol.NVALUES, expand.n), dtype=np.float32)

    def key(self, actions, symbol):
        return self.expand(self.base.key(actions, symbol))

    def write(self, actions, symbol, label):
        k = self.key(actions, symbol)
        target = np.zeros(sol.NVALUES)
        target[int(label)] = 1.0
        error = target - self.w @ k if self.writer == "delta" else target
        self.w += np.float32(sol.BETA) * np.outer(error, k).astype(np.float32)

    def predict(self, actions, symbol):
        return int(np.argmax(self.w @ self.key(actions, symbol)))

    @property
    def byte_count(self):
        return self.w.nbytes


def run_world(seed: int, nvisited: int, models: dict) -> dict:
    """Sol's sol.run_world protocol, line for line, with a caller-supplied model set."""
    places, symbols, values, order = sol.make_world(seed)
    visited = order[:nvisited]
    shuffled = values.copy()
    rng = np.random.default_rng(seed + 70000)
    shuffled[visited] = rng.permutation(values[visited])

    def write(indices, current):
        for i in indices:
            path = sol.canonical_path(places[i])
            for name, m in models.items():
                y = shuffled[i] if name == "shuffled_labels" else current[i]
                m.write(path, int(symbols[i]), int(y))

    for _ in range(3):
        write(visited, values)

    def score(indices, target):
        return {name: sol.evaluate(m, indices, places, symbols, target, sol.different_path)
                for name, m in models.items()}

    before = score(visited, values)
    rng = np.random.default_rng(seed + 50000)
    rewritten = np.sort(rng.choice(visited, size=max(1, len(visited) // 4), replace=False))
    untouched = np.setdiff1d(visited, rewritten)
    next_values = values.copy()
    for i in rewritten:
        alt = int(rng.integers(0, sol.NVALUES - 1))
        next_values[i] = (int(values[i]) + 1 + alt) % sol.NVALUES
    for _ in range(3):
        write(rewritten, next_values)
    return {"first_new_route": before, "rewritten": score(rewritten, next_values),
            "untouched": score(untouched, values)}


def make_models(n: int, k: int) -> dict:
    return {
        "sol_phase_delta": sol.DeltaMemory("phase"),
        "expanded_phase_delta": ExpandedMemory("phase", Expansion(n, k)),
        "linear_wide_phase_delta": ExpandedMemory("phase", Expansion(n, k, linear=True)),
        "expanded_content_delta": ExpandedMemory("content", Expansion(n, k)),
        "expanded_phase_hebb": ExpandedMemory("phase", Expansion(n, k), writer="hebb"),
        "indexed_coordinate": sol.IndexedCoordinate(),
    }


def bench(n: int, k: int, seeds=SEEDS) -> dict:
    out = {}
    for nv in (18, 30):
        cases = [run_world(s, nv, make_models(n, k)) for s in seeds]
        out[str(nv)] = {phase: {m: float(np.mean([c[phase][m] for c in cases])) for m in cases[0][phase]}
                        for phase in ("first_new_route", "rewritten", "untouched")}
    return out


def key_geometry(n: int, k: int) -> dict:
    E = Expansion(n, k)
    G = np.array([sol.PhaseIntegrator().follow(sol.canonical_path(p)) for p in
                  [(x, y) for x in range(sol.SIDE) for y in range(sol.SIDE)]])
    P = np.array([E(g) for g in G])
    off = ~np.eye(len(G), dtype=bool)
    cg, cp = (G @ G.T)[off], (P @ P.T)[off]
    return {"phase_mean_cos": float(cg.mean()), "phase_max_cos": float(cg.max()),
            "expanded_mean_cos": float(cp.mean()), "expanded_max_cos": float(cp.max())}


def decide(r: dict) -> dict:
    a, b = r["18"], r["30"]
    e = "expanded_phase_delta"
    g = {
        "S1_new_route_18>=0.95": a["first_new_route"][e] >= 0.95,
        "S2_new_route_30>=0.90": b["first_new_route"][e] >= 0.90,
        "S3_untouched_18>=0.90_and_30>=0.85": a["untouched"][e] >= 0.90 and b["untouched"][e] >= 0.85,
        "S4_rewritten_18_and_30>=0.95": a["rewritten"][e] >= 0.95 and b["rewritten"][e] >= 0.95,
        "S5_linear_within_5pts_of_sol_at_18":
            abs(a["first_new_route"]["linear_wide_phase_delta"] - a["first_new_route"]["sol_phase_delta"]) <= 0.05,
        "S6_expanded_content_18<=0.45": a["first_new_route"]["expanded_content_delta"] <= 0.45,
    }
    g = {k: bool(v) for k, v in g.items()}
    return {"gates": g, "decision": "PASS" if all(g.values()) else "FAIL"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sharp_address.json")
    args = ap.parse_args()
    n, k = PRIMARY
    primary = bench(n, k)
    res = {"vendored_commit": "0c234a3", "primary": {"N": n, "k": k, "results": primary,
           "key_geometry": key_geometry(n, k), **decide(primary)},
           "bytes": {"expanded_memory": sol.NVALUES * n * 4, "W_float32": n * sol.DIM * 4,
                     "sol_phase_memory": sol.MEMORY_BYTES},
           "sweep": [{"N": sn, "k": sk, "results": bench(sn, sk), "key_geometry": key_geometry(sn, sk)}
                     for sn, sk in SWEEP]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    print(f"primary N={n} k={k}: {res['primary']['decision']}")
    for gname, ok in res["primary"]["gates"].items():
        print(f"  {'pass' if ok else 'FAIL'}  {gname}")
    print("  key geometry:", {kk: round(v, 3) for kk, v in res["primary"]["key_geometry"].items()})
    for nv in ("18", "30"):
        print(f"\n visited {nv}: model            new_route  rewritten  untouched")
        for m in primary[nv]["first_new_route"]:
            print(f"  {m:26s} {primary[nv]['first_new_route'][m]:.3f}      "
                  f"{primary[nv]['rewritten'][m]:.3f}      {primary[nv]['untouched'][m]:.3f}")
    print("\n sweep (expanded_phase_delta, new route 18 / 30, untouched 18 / 30, max key cos)")
    for s in res["sweep"]:
        r = s["results"]
        e = "expanded_phase_delta"
        print(f"  N={s['N']:5d} k={s['k']:3d}   {r['18']['first_new_route'][e]:.3f} / {r['30']['first_new_route'][e]:.3f}"
              f"   {r['18']['untouched'][e]:.3f} / {r['30']['untouched'][e]:.3f}   {s['key_geometry']['expanded_max_cos']:.3f}")


if __name__ == "__main__":
    main()
