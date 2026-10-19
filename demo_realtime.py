"""Gate 5: real-time credit vs a three-factor Hebbian rule.  -> runs/realtime/

A  bias: alignment of the TRIAL-AVERAGED update with the true gradient
B  variance: alignment of SINGLE-TRIAL updates
C  training on amplitude-jittered patterns, evaluated on held-out jitter draws
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tuft_gate import fresh, train_exact, task, true_descent
from realtime_gate import RealTime, updates, train

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
SEEDS = 4
TRUNK = 8
ALIGN_RULES = [("ping-local", "ping-local", {}),
               ("three-factor", "three-factor", {}),
               ("ping-local, bursts off", "ping-local", dict(bursts=False))]
TRAIN_CONDS = [("exact gradient (bound)", "exact", {}),
               ("ping-local", "ping-local", {}),
               ("three-factor", "three-factor", {}),
               ("ping-local, bursts off", "ping-local", dict(bursts=False)),
               ("ping-local, shuffled context", "ping-local", dict(shuffle=True)),
               ("three-factor, shuffled context", "three-factor", dict(shuffle=True))]


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def alignment_stats(trials):
    xs, ys = task()
    out = {name: {"mean_update": [], "single_trial": []} for name, _, _ in ALIGN_RULES}
    for s in range(SEEDS):
        t = fresh(s, trunk=TRUNK)
        train_exact(t, xs, ys, 30, seed=s)              # probe state: mixed error signs
        truth = np.concatenate([true_descent(t, x, y)[0] for x, y in zip(xs, ys)])
        for name, rule, kw in ALIGN_RULES:
            rt = RealTime(t, **kw)
            rng = np.random.default_rng(50 + s)
            per = np.zeros((trials, 6, 4))
            for k in range(trials):
                for j, (x, y) in enumerate(zip(xs, ys)):
                    rt.burst_theta = None
                    per[k, j] = updates(rt, x, y, rng, rule)[0]
            out[name]["mean_update"].append(cos(per.mean(0).ravel(), truth))
            out[name]["single_trial"].append(float(np.mean([cos(per[k].ravel(), truth)
                                                            for k in range(trials)])))
        print("align seed", s, {n: (round(v["mean_update"][-1], 3), round(v["single_trial"][-1], 3))
                                for n, v in out.items()}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("runs/realtime"))
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.replot:
        plot(args, json.loads((args.output / "metrics.json").read_text()))
        return
    m = {"trunk": TRUNK, "held_out_jitter": "+-5% per active input", "epochs": args.epochs,
         "lr": 0.02, "units": "dimensionless"}
    m["A_B_alignment"] = alignment_stats(args.trials)
    res = {}
    for name, rule, kw in TRAIN_CONDS:
        rows = []
        for s in range(SEEDS):
            (acc, margin, loss), _ = train(s, rule, trunk=TRUNK, epochs=args.epochs, lr=0.02, **kw)
            rows.append({"seed": s, "held_out_accuracy": acc, "margin": margin, "loss": loss})
        res[name] = rows
        print(name, [(round(r["held_out_accuracy"], 3), round(r["margin"], 3)) for r in rows], flush=True)
    m["C_training"] = res
    (args.output / "metrics.json").write_text(json.dumps(m, indent=2))
    plot(args, m)


def plot(args, m):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axs.ravel()
    al = m["A_B_alignment"]
    names = list(al)
    for k, key, title in ((0, "mean_update", "A. Bias: trial-averaged update"),
                          (1, "single_trial", "B. Variance: single-trial update")):
        a = ax[k]
        for i, n in enumerate(names):
            v = al[n][key]
            a.scatter(v, [i] * len(v), color=BLUE if np.mean(v) > 0.8 else ORANGE, s=24, zorder=3)
        a.set_yticks(range(len(names)), names)
        a.invert_yaxis()
        a.axvline(0, color=MUTED, lw=0.8)
        a.set_xlim(-1.05, 1.05)
        a.set_xlabel("alignment with true gradient (4 probe seeds)")
        a.set_title(title, loc="left", color=INK)
    a = ax[2]
    C = m["C_training"]
    tn = list(C)
    for i, n in enumerate(tn):
        accs = [r["held_out_accuracy"] for r in C[n]]
        a.scatter(accs, [i] * len(accs), color=BLUE if np.mean(accs) >= 0.9 else ORANGE, s=24, zorder=3)
        a.text(1.02, i, f"mean {np.mean(accs):.2f}", transform=a.get_yaxis_transform(),
               va="center", color=MUTED)
    if "C_cluster_calibrate_only" in m:
        cal = [r["acc"] for r in m["C_cluster_calibrate_only"]]
        a.axvspan(min(cal), max(cal), color=ORANGE, alpha=0.12)
        a.text(np.mean(cal), 2.5, "bias-only\ntraining", color=MUTED, ha="center", va="center")
    a.set_yticks(range(len(tn)), tn)
    a.invert_yaxis()
    a.set_xlim(0, 1.05)
    a.set_xlabel("held-out accuracy (+-5% amplitude jitter, 30 patterns)")
    a.set_title("C. Cluster task: shuffled labels do as well", loc="left", color=INK)

    a = ax[3]
    if "D_side_task" in m:
        D = m["D_side_task"]
        dn = list(D)
        for i, n in enumerate(dn):
            accs = [r["acc"] for r in D[n]]
            a.scatter(accs, [i] * len(accs), color=BLUE if np.mean(accs) >= 0.9 else ORANGE, s=24, zorder=3)
            a.text(1.02, i, f"mean {np.mean(accs):.2f}", transform=a.get_yaxis_transform(),
                   va="center", color=MUTED)
        a.axvline(0.5, color=MUTED, lw=0.8, ls="--")
        a.set_yticks(range(len(dn)), dn)
        a.invert_yaxis()
        a.set_xlim(0, 1.05)
        a.set_xlabel("held-out accuracy (left cluster high / right cluster low)")
        a.set_title("D. A task the cell cannot solve by itself", loc="left", color=INK)
    fig.savefig(args.output / "realtime.png", dpi=150)


if __name__ == "__main__":
    main()
