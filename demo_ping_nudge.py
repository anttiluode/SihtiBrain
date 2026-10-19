"""Second gate: what a ping can do once the dendrite is nonlinear.

Produces runs/ping_nudge/{metrics.json, ping_nudge.png}.

  A  observability   which cable modes the soma can see at all
  B  ping as nudge   EqProp gradient from a somatic error current vs exact
  C  learning        cluster detection, nonlinear vs linear tree
  D  ping as kick    matched-soma inputs, identical ping, persistent plateau
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from ping_nudge import (Tree, task, pattern, linear_floor, train, evaluate,
                        eqprop_grad, finite_difference_grad, soma_observability,
                        sihti_residues)

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"


def margin(outs):
    return float(outs[:2].min() - outs[2:].max())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("runs/ping_nudge"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    m = {"units": "dimensionless"}

    # A. observability ----------------------------------------------------
    lam, phi, w0 = soma_observability(Tree())
    m["A_modes_eigenvalues"] = lam.round(6).tolist()
    m["A_mode_soma_weight"] = w0.round(6).tolist()
    m["A_invisible_modes"] = int(np.sum(np.abs(w0) < 1e-10))

    # B. gradient check ---------------------------------------------------
    gt = Tree(a=0.27, k=6, theta=1.1, w=[0.9, 0.7, 0.8, 0.6], b=-0.3)
    xs, ys = task(hi=0.6, lo=0.3)
    fd = finite_difference_grad(gt, xs[0], ys[0])
    betas = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    errs, coss = [], []
    for b in betas:
        ge, *_ = eqprop_grad(gt, xs[0], ys[0], b)
        errs.append(float(np.linalg.norm(ge - fd) / np.linalg.norm(fd)))
        coss.append(float(ge @ fd / np.linalg.norm(ge) / np.linalg.norm(fd)))
    slope = float(np.polyfit(np.log(betas[1:]), np.log(errs[1:]), 1)[0])
    m["B_beta"] = betas
    m["B_rel_error_vs_finite_difference"] = errs
    m["B_cosine"] = coss
    m["B_loglog_slope"] = slope

    # C. learning ---------------------------------------------------------
    floor, _ = linear_floor(xs, ys)
    m["C_linear_floor_any_passive_tree"] = floor
    runs = {}
    for label, a in (("nonlinear", 0.27), ("linear", 0.0)):
        runs[label] = []
        for seed in range(args.seeds):
            r = np.random.default_rng(100 + seed)
            t = Tree(a=a, k=6, theta=1.1, w=r.uniform(0.3, 1.2, 4), b=r.uniform(-1, 0))
            hist = train(t, xs, ys, beta=0.02, lr=0.05, epochs=args.epochs,
                         train_g=False, train_a=False, seed=seed, record_every=5)
            loss, outs, correct = evaluate(t, xs, ys)
            runs[label].append({"seed": seed, "history": hist, "loss": loss,
                                "outputs": outs.round(5).tolist(),
                                "margin": margin(outs), "correct_of_6": correct,
                                "w": t.w.round(4).tolist(), "b": round(t.b, 4)})
    m["C_runs"] = {k: [{kk: vv for kk, vv in d.items() if kk != "history"} for d in v]
                   for k, v in runs.items()}

    # D. kick ---------------------------------------------------------------
    kt = Tree(a=1.0, k=6, theta=2.0)
    xc = pattern((2, 3), 0.5)
    vc = kt.relax(xc, steps=4000)
    a_split = brentq(lambda A: kt.relax(pattern((2, 5), A), steps=4000)[0] - vc[0],
                     0.3, 1.0, xtol=1e-12)
    xsp = pattern((2, 5), a_split)
    vs = kt.relax(xsp, steps=4000)

    def end_state(x, v0, amp):
        return kt.trajectory(x, steps=6000, dt=0.02, v0=v0, pulse=(100, 150, amp))[0][-1]

    def flip_amp(x, v0, lo, hi, level):
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if end_state(x, v0, mid)[0] > level else (mid, hi)
        return hi

    c_flip = flip_amp(xc, vc, 0, 10, 1.0)
    s_flip = flip_amp(xsp, vs, 0, 12, 1.0)
    ping = 5.0
    tr_c, e_c = kt.trajectory(xc, steps=6000, dt=0.02, v0=vc, pulse=(100, 150, ping))
    tr_s, e_s = kt.trajectory(xsp, steps=6000, dt=0.02, v0=vs, pulse=(100, 150, ping))
    lin = Tree(a=0.0, k=6, theta=2.0)
    xl = pattern((2, 5), 0.5)
    tr_lc, _ = lin.trajectory(xc, steps=6000, dt=0.02, v0=lin.relax(xc, steps=4000), pulse=(100, 150, ping))
    tr_ls, _ = lin.trajectory(xl, steps=6000, dt=0.02, v0=lin.relax(xl, steps=4000), pulse=(100, 150, ping))

    basins = {}
    rng = np.random.default_rng(0)
    for name, x in (("cluster", xc), ("split", xsp)):
        found = set()
        for _ in range(200):
            v = kt.relax(x, v0=rng.uniform(-1, 5, 7), steps=3000)
            found.add(tuple(np.round(v[[0, 1, 4]], 4)))
        basins[name] = sorted(found)
    post = (e_c[150:], e_s[150:])
    m["D"] = {
        "cluster_amp": 0.5, "split_amp_matched": a_split,
        "pre_ping_soma": [float(vc[0]), float(vs[0])],
        "pre_ping_roots_cluster": vc[[1, 4]].round(5).tolist(),
        "pre_ping_roots_split": vs[[1, 4]].round(5).tolist(),
        "ping_amplitude": ping, "ping_steps": [100, 150],
        "post_ping_soma": [float(tr_c[-1, 0]), float(tr_s[-1, 0])],
        "cluster_flips_at_amplitude": c_flip, "split_flips_at_amplitude": s_flip,
        "basins_soma_root1_root4": {k: [list(map(float, b)) for b in v] for k, v in basins.items()},
        "energy_max_upward_step_after_ping": float(max(np.diff(post[0]).max(), np.diff(post[1]).max())),
        "linear_control_post_ping_soma_abs_difference": float(np.abs(tr_lc[:, 0] - tr_ls[:, 0]).max()),
    }

    # E. Sihti identity ------------------------------------------------------
    h = Tree().laplacian() + 0.15 * np.eye(7)
    x0 = np.random.default_rng(1).normal(size=7)
    res, path = sihti_residues(h, x0, 0.05, 200)
    m["E_sihti_reconstruction_error"] = float(np.abs(res.sum(0) + path[-1] - x0).max())

    (args.output / "metrics.json").write_text(json.dumps(m, indent=2))

    # ---- figure -------------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    a0 = ax[0, 0]
    cols = [GRAY if abs(w) < 1e-10 else BLUE for w in w0]
    a0.bar(range(7), np.abs(w0), color=cols, width=0.6)
    a0.set_xticks(range(7), [f"{l:.2f}" for l in lam])
    a0.set_xlabel("cable mode (decay rate)")
    a0.set_ylabel("|mode amplitude at soma|")
    a0.set_title("A. The soma sees 3 of 7 modes; 4 are exactly invisible", loc="left", color=INK)
    for i, w in enumerate(w0):
        if abs(w) < 1e-10:
            a0.text(i, 0.02, "0", ha="center", color=MUTED)

    a1 = ax[0, 1]
    a1.loglog(betas, errs, "o-", color=BLUE, lw=2, ms=6)
    a1.set_xlabel("nudge strength beta (somatic error current gain)")
    a1.set_ylabel("relative error vs exact gradient")
    a1.set_title(f"B. Error ping gives the exact gradient (slope {slope:.2f}, cos -> 1)", loc="left", color=INK)

    a2 = ax[1, 0]
    for label, color in (("nonlinear", BLUE), ("linear", GRAY)):
        for i, run in enumerate(runs[label]):
            ep, ls = zip(*run["history"])
            a2.semilogy(ep, ls, color=color, lw=1.6, alpha=0.9,
                        label=f"{label} branches ({run['correct_of_6']}/6)" if i == 0 else None)
    a2.axhline(floor, color=ORANGE, lw=1.2, ls="--", label="proven floor for any passive tree")
    a2.set_xlabel("epoch")
    a2.set_ylabel("mean somatic loss")
    a2.set_title("C. Synapses learn from the ping; linear tree is stuck at its floor", loc="left", color=INK)
    a2.legend(frameon=False)

    a3 = ax[1, 1]
    tt = 0.02 * np.arange(len(tr_c))
    a3.plot(tt, tr_lc[:, 0], color=GRAY, lw=3, alpha=0.5, label="linear tree, both inputs (identical)")
    a3.plot(tt, tr_c[:, 0], color=BLUE, lw=2, label="clustered input")
    a3.plot(tt, tr_s[:, 0], color=ORANGE, lw=2, ls="--", label="split input")
    a3.annotate("same ping at t = 2", xy=(3.0, 3.3), xytext=(12, 3.3), color=MUTED,
                    arrowprops=dict(arrowstyle="->", color=MUTED), va="center")
    a3.set_xlim(-2, 60)
    a3.set_xlabel("time")
    a3.set_ylabel("somatic voltage")
    a3.set_title("D. Same soma, same ping: only the clustered branch tips over", loc="left", color=INK)
    a3.legend(frameon=False, loc="center right")
    fig.savefig(args.output / "ping_nudge.png", dpi=150)
    print(json.dumps({k: v for k, v in m.items() if k != "C_runs"}, indent=1)[:3000])
    for k, v in m["C_runs"].items():
        for d in v:
            print(k, d["seed"], f"loss {d['loss']:.5f} margin {d['margin']:+.4f} {d['correct_of_6']}/6")


if __name__ == "__main__":
    main()
