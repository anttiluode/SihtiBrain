"""Gate 4: context at the tuft.  -> runs/tuft/{metrics.json, tuft.png}

A  how much apical context reaches the soma vs trunk length (passive, and the
   best single near-critical hotspot that stays monostable)
B  alignment of the local two-ping update vs trunk length, under calcium read noise
C  alignment by context code on an 8-compartment trunk
D  training by context code on an 8-compartment trunk with 1 % read noise
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tuft_gate import (task, fresh, train_exact, attenuation, alignment, Readout,
                       train_local)

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
SEEDS = 4
CODES = [("graded at soma (ideal)", "soma", 0.2, 0.0),
         ("graded at tuft, passive trunk", "tuft", 0.2, 0.0),
         ("ternary event: burst / none / pause", "ternary", 0.05, 0.06),
         ("one-sided burst only", "burst", 0.05, 0.06)]


def probes(trunk):
    xs, ys = task()
    out = []
    for s in range(SEEDS):
        t = fresh(s, trunk=trunk)
        train_exact(t, xs, ys, 30, seed=s)
        out.append(t)
    return out


def best_hotspot_alpha(trunk, x):
    best = 0.0
    for th in (0.3, 0.6, 1.0):
        for a in np.linspace(0.0, 1.0, 41):
            t = fresh(0, trunk=trunk, a_hot=a, theta_hot=th)
            lo = t.relax(x)
            hi = t.relax(x, v0=np.full(t.n, 3.0))
            if abs(lo[0] - hi[0]) > 1e-4:          # bistable: no longer a graded amplifier
                continue
            best = max(best, attenuation(t, x)[0])
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("runs/tuft"))
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    xs, ys = task()
    m = {"units": "dimensionless", "probe": "30 epochs of exact gradient per seed",
         "context_strength_beta_graded": 0.2, "event_size": 0.05, "event_dead_zone": 0.06}

    # A ---------------------------------------------------------------------------
    lengths = [0, 1, 2, 3, 4, 6, 8, 12, 16]
    passive, resid, hot = [], [], []
    for L in lengths:
        a, r = attenuation(fresh(0, trunk=L), xs[0])
        passive.append(a); resid.append(r)
        hot.append(best_hotspot_alpha(L, xs[0]) if L >= 1 else a)
    m["A_lengths"] = lengths
    m["A_alpha_passive"] = passive
    m["A_alpha_best_monostable_hotspot"] = hot
    m["A_cut_vertex_max_residual"] = float(max(resid))
    print("A", [round(p, 4) for p in passive], [round(h, 4) for h in hot], max(resid), flush=True)

    # B ---------------------------------------------------------------------------
    b_lengths = [0, 2, 4, 6, 8]
    noises = [0.0, 0.01, 0.03]
    B = {}
    for L in b_lengths:
        ps = probes(L)
        for sig in noises:
            B[f"{L}|{sig}"] = [alignment(t, Readout(noise=sig), xs, ys, 0.2, "tuft",
                                         np.random.default_rng(10 + i)) for i, t in enumerate(ps)]
    m["B_alignment_tuft_graded"] = B
    print("B done", flush=True)

    # C ---------------------------------------------------------------------------
    ps8 = probes(8)
    C = {}
    for name, where, beta, dead in CODES:
        for sig in noises:
            C[f"{name}|{sig}"] = [alignment(t, Readout(noise=sig), xs, ys, beta, where,
                                            np.random.default_rng(10 + i), dead)
                                  for i, t in enumerate(ps8)]
    m["C_alignment_by_code_trunk8"] = C
    print("C done", flush=True)

    # D ---------------------------------------------------------------------------
    D = {}
    for name, where, beta, dead in CODES:
        res = []
        for s in range(SEEDS):
            t = fresh(s, trunk=8)
            loss, outs, correct = train_local(t, Readout(noise=0.01), xs, ys, beta, where,
                                              epochs=args.epochs, seed=s, dead=dead)
            res.append({"seed": s, "loss": loss, "correct_of_6": correct,
                        "margin": float(outs[:2].min() - outs[2:].max()),
                        "outputs": outs.round(4).tolist()})
        D[name] = res
        print(name, [(r["correct_of_6"], round(r["margin"], 3)) for r in res], flush=True)
    m["D_training_trunk8_noise1pct"] = D
    (args.output / "metrics.json").write_text(json.dumps(m, indent=2))
    plot(args, m)


def plot(args, m):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    a = ax[0, 0]
    L = m["A_lengths"]
    a.semilogy(L, m["A_alpha_passive"], "o-", color=GRAY, lw=2, label="passive trunk")
    a.semilogy(L, m["A_alpha_best_monostable_hotspot"], "o-", color=BLUE, lw=2,
               label="best single hotspot, still monostable")
    a.set_xlabel("trunk length (compartments between soma and tuft)")
    a.set_ylabel("context reaching the soma, alpha")
    a.set_title("A. Apical context decays exponentially; one hotspot buys ~3x", loc="left", color=INK)
    a.legend(frameon=False)

    a = ax[0, 1]
    B = m["B_alignment_tuft_graded"]
    lens = sorted({int(k.split("|")[0]) for k in B})
    for sig, col in ((0.0, GRAY), (0.01, BLUE), (0.03, ORANGE)):
        vals = np.array([B[f"{l}|{sig}"] for l in lens])
        a.plot(lens, vals.mean(1), "o-", color=col, lw=2, label=f"Ca read noise {sig:.0%}")
        a.fill_between(lens, vals.min(1), vals.max(1), color=col, alpha=0.12)
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xlabel("trunk length")
    a.set_ylabel("alignment with true gradient")
    a.set_title("B. Direction is exact; noise drowns it as the trunk grows", loc="left", color=INK)
    a.legend(frameon=False, loc="lower left")

    a = ax[1, 0]
    C = m["C_alignment_by_code_trunk8"]
    names = [c[0] for c in CODES]
    for j, (sig, col) in enumerate(((0.0, GRAY), (0.01, BLUE), (0.03, ORANGE))):
        for i, n in enumerate(names):
            v = C[f"{n}|{sig}"]
            a.scatter(v, [i + (j - 1) * 0.18] * len(v), color=col, s=16,
                      label=f"noise {sig:.0%}" if i == 0 else None)
    a.set_yticks(range(len(names)), names)
    a.invert_yaxis()
    a.axvline(0, color=MUTED, lw=0.8)
    a.set_xlim(-0.35, 1.05)
    a.set_xlabel("alignment with true gradient (8-compartment trunk, 4 seeds)")
    a.set_title("C. An event code keeps the credit; passive spread loses it", loc="left", color=INK)
    a.legend(frameon=False, loc="lower left")

    a = ax[1, 1]
    D = m["D_training_trunk8_noise1pct"]
    for i, n in enumerate(names):
        ms = [r["margin"] for r in D[n]]
        sep = sum(x > 0 for x in ms)
        cal = sum(r["correct_of_6"] == 6 for r in D[n])
        a.scatter(ms, [i] * len(ms), color=BLUE if sep == len(ms) else ORANGE, s=22, zorder=3)
        a.text(1.02, i, f"{sep}/{len(ms)} separate, {cal}/{len(ms)} at 6/6",
               transform=a.get_yaxis_transform(), va="center", color=MUTED)
    a.set_yticks(range(len(names)), names)
    a.invert_yaxis()
    a.axvline(0, color=MUTED, lw=0.8)
    a.set_xlabel("somatic margin after training (cluster min - split max)")
    a.set_title("D. Training with 1% read noise, 8-compartment trunk", loc="left", color=INK)
    fig.savefig(args.output / "tuft.png", dpi=150)


if __name__ == "__main__":
    main()
