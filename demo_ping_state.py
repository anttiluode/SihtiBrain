"""Gate 3: the ping with state, up and down.  -> runs/ping_state/

Alignment is measured at 'probe states' (each seed's tree after 30 epochs of
exact EqProp), where the six patterns' errors have mixed signs; at the initial
state every output overshoots every target and even shuffled labels align.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ping_nudge import task, train, LEAVES, SomaNudge
from ping_state import (PingReadout, alignment, train_ping, fresh_tree, _sigmoid)

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
SEEDS = 4
BASE = dict(ca_thr=5.0, ca_k=4.0)
GAMMA10 = np.exp(np.random.default_rng(1).uniform(-np.log(10), np.log(10), 4))


def probe_trees(xs, ys):
    trees = []
    for s in range(SEEDS):
        t = fresh_tree(s)
        train(t, xs, ys, beta=0.02, lr=0.05, epochs=30, train_g=False, train_a=False, seed=s)
        trees.append(t)
    return trees


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("runs/ping_state"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--replot", action="store_true",
                    help="redraw from an existing metrics.json; recompute only panel D")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    xs, ys = task(hi=0.6, lo=0.3)
    trees = probe_trees(xs, ys)
    if args.replot:
        m = json.loads((args.output / "metrics.json").read_text())
        thrs = np.array(m["A_thresholds"])
        leafv = np.array(m["A_operating_leaf_voltage_range"])
        sign_law = {float(a): np.array(v) for a, v in m["A_alignment_by_threshold"].items()}
        robust = m["B_alignment"]
        trained = m["C_training"]
    else:
        m, thrs, leafv, sign_law, robust, trained = compute(args, xs, ys, trees)
    plot(args, m, xs, ys, trees, thrs, leafv, sign_law, robust, trained)


def compute(args, xs, ys, trees):
    m = {"units": "dimensionless", "probe": "30 epochs of exact EqProp per seed"}

    def al(rd, beta=0.02, noise_seed=None, **kw):
        out = []
        for s, t in enumerate(trees):
            rng = None if noise_seed is None else np.random.default_rng(noise_seed + s)
            out.append(alignment(t, rd, xs, ys, beta=beta, rng=rng, **kw))
        return out

    # A. the sign law ------------------------------------------------------
    thrs = np.linspace(0.25, 6.0, 24)
    leafv = np.concatenate([[t.relax(x)[list(LEAVES)] for x in xs] for t in trees]).ravel()
    sign_law = {}
    for amp in (2.0, 32.0):
        sign_law[amp] = np.array([al(PingReadout(amp=amp, ca_thr=th, ca_k=4.0)) for th in thrs])
    m["A_thresholds"] = thrs.round(4).tolist()
    m["A_operating_leaf_voltage_range"] = [float(leafv.min()), float(leafv.max())]
    m["A_alignment_by_threshold"] = {str(a): v.round(4).tolist() for a, v in sign_law.items()}

    # B. robustness of the credit -------------------------------------------
    conds = [
        ("ping, fractional Ca", dict(rd=PingReadout(**BASE))),
        ("ping, absolute Ca", dict(rd=PingReadout(**BASE), compare="diff")),
        ("ping, synapse past unblock midpoint", dict(rd=PingReadout(ca_thr=0.5, ca_k=4.0))),
        ("bAP attenuation x10 non-reciprocal", dict(rd=PingReadout(gamma=GAMMA10, **BASE))),
        ("  ... with a huge ping (amp 32)", dict(rd=PingReadout(amp=32.0, gamma=GAMMA10, **BASE))),
        ("waveform mod 0.25, Brunner sign flip", dict(rd=PingReadout(mod=[-.25, .25, -.25, .25], **BASE))),
        ("waveform mod 1, Brunner sign flip", dict(rd=PingReadout(mod=[-1, 1, -1, 1], **BASE))),
        ("waveform mod 1, uniform +", dict(rd=PingReadout(mod=1.0, **BASE))),
        ("Ca read noise 3%, teacher 0.02", dict(rd=PingReadout(noise=0.03, **BASE), noise_seed=10)),
        ("Ca read noise 10%, teacher 0.2", dict(rd=PingReadout(noise=0.10, **BASE), noise_seed=10, beta=0.2)),
        ("shuffled teacher", dict(rd=PingReadout(**BASE), shuffle=True)),
        ("no ping", dict(rd=PingReadout(**BASE), use_ping=False)),
    ]
    robust = {}
    for name, c in conds:
        c = dict(c); rd = c.pop("rd")
        robust[name] = al(rd, **c)
    m["B_alignment"] = {k: np.round(v, 4).tolist() for k, v in robust.items()}

    # C. training -------------------------------------------------------------
    train_conds = [
        ("exact EqProp nudge (gate 2)", dict(rule="eqprop", rd=PingReadout(**BASE))),
        ("ping, fractional Ca", dict(rd=PingReadout(**BASE))),
        ("ping, absolute Ca", dict(rd=PingReadout(**BASE), compare="diff")),
        ("ping, past unblock midpoint", dict(rd=PingReadout(ca_thr=0.5, ca_k=4.0))),
        ("bAP attenuation x10", dict(rd=PingReadout(gamma=GAMMA10, **BASE))),
        ("waveform mod 1, Brunner flip", dict(rd=PingReadout(mod=[-1, 1, -1, 1], **BASE))),
        ("shuffled teacher", dict(rd=PingReadout(**BASE), shuffle=True)),
        ("no ping", dict(rd=PingReadout(**BASE), rule="noping")),
    ]
    trained = {}
    for name, c in train_conds:
        c = dict(c); rd = c.pop("rd")
        res = []
        for s in range(SEEDS):
            t = fresh_tree(s)
            loss, outs, correct = train_ping(t, rd, xs, ys, epochs=args.epochs, seed=s, **c)
            res.append({"seed": s, "loss": loss, "margin": float(outs[:2].min() - outs[2:].max()),
                        "correct_of_6": correct, "outputs": outs.round(4).tolist()})
        trained[name] = res
        print(name, [(r["correct_of_6"], round(r["margin"], 3)) for r in res], flush=True)
    m["C_training"] = trained
    return m, thrs, leafv, sign_law, robust, trained


def measured_vs_needed(trees, xs, ys, beta=0.02, linear=False):
    """For every active synapse: (dv_i the EqProp update needs,
    log(Ca_teach/Ca_free)/k the synapse actually measures)."""
    from ping_nudge import Tree
    rd = PingReadout(**BASE)
    need, got = [], []
    for t in trees:
        if linear:
            t = Tree(a=0.0, w=t.w, b=t.b)
        for x, y in zip(xs, ys):
            v0 = t.relax(x)
            vb = t.relax(x, v0=v0, nudge=SomaNudge(beta, y))
            ca0, cab = rd.calcium(t, v0, x), rd.calcium(t, vb, x)
            act = np.asarray(x) > 0
            need += list((vb[list(LEAVES)] - v0[list(LEAVES)])[act])
            got += list((np.log(cab[act] / ca0[act])) / rd.ca_k)
    return np.array(need), np.array(got)


def plot(args, m, xs, ys, trees, thrs, leafv, sign_law, robust, trained):
    # D. what each synapse measures vs what EqProp needs ----------------------
    need_nl, got_nl = measured_vs_needed(trees, xs, ys)
    need_li, got_li = measured_vs_needed(trees, xs, ys, linear=True)
    rel = lambda a, b: float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
    m["D"] = {"beta": 0.02, "n_synapse_samples": int(len(need_nl)),
              "nonlinear_max_rel_dev": rel(need_nl, got_nl),
              "nonlinear_corr": float(np.corrcoef(need_nl, got_nl)[0, 1]),
              "linear_max_rel_dev": rel(need_li, got_li)}
    (args.output / "metrics.json").write_text(json.dumps(m, indent=2))

    # ---- figure -------------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    a = ax[0, 0]
    a.axvspan(leafv.min(), leafv.max(), color=GRAY, alpha=0.15)
    a.text(leafv.min() + 0.08, 0.55, "operating\nleaf voltages", ha="left", color=MUTED)
    for amp, col in ((2.0, BLUE), (32.0, ORANGE)):
        v = sign_law[amp]
        a.plot(thrs, v.mean(1), color=col, lw=2, label=f"ping amplitude {amp:g}")
        a.fill_between(thrs, v.min(1), v.max(1), color=col, alpha=0.15)
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xlabel("calcium-unblock midpoint (local voltage)")
    a.set_ylabel("alignment with true gradient")
    a.set_title("A. Credit sign follows the curvature of the Ca readout", loc="left", color=INK)
    a.legend(frameon=False, loc="lower right")

    a = ax[0, 1]
    names = list(robust)
    for i, n in enumerate(names):
        vals = robust[n]
        a.scatter(vals, [i] * len(vals), color=BLUE if np.mean(vals) > 0.9 else ORANGE, s=22, zorder=3)
    a.set_yticks(range(len(names)), names)
    a.invert_yaxis()
    a.axvline(0, color=MUTED, lw=0.8)
    a.set_xlim(-1.05, 1.05)
    a.set_xlabel("alignment with true gradient (4 seeds)")
    a.set_title("B. What the credit survives, and what breaks it", loc="left", color=INK)

    a = ax[1, 0]
    tn = list(trained)
    for i, n in enumerate(tn):
        ms = [r["margin"] for r in trained[n]]
        ok = all(r["correct_of_6"] == 6 for r in trained[n])
        a.scatter(ms, [i] * len(ms), color=BLUE if ok else ORANGE, s=22, zorder=3)
        a.text(1.02, i, f"{sum(r['correct_of_6'] == 6 for r in trained[n])}/{SEEDS} solve",
               transform=a.get_yaxis_transform(), va="center", color=MUTED)
    a.set_yticks(range(len(tn)), tn)
    a.invert_yaxis()
    a.axvline(0, color=MUTED, lw=0.8)
    a.set_xlabel("somatic margin after training (cluster min - split max)")
    a.set_title("C. Two pings and a teacher train the tree; no error signal", loc="left", color=INK)

    a = ax[1, 1]
    lim = 1.08 * max(np.abs(need_nl).max(), np.abs(got_nl).max())
    a.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1, ls="--", label="exact credit")
    a.scatter(need_li, got_li, s=34, facecolors="none", edgecolors=GRAY, lw=1.2,
              label=f"linear tree (max dev {m['D']['linear_max_rel_dev']:.1e})")
    a.scatter(need_nl, got_nl, s=14, color=BLUE,
              label=f"nonlinear tree (r = {m['D']['nonlinear_corr']:.4f})")
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_xlabel("voltage change the exact update needs, dv_i")
    a.set_ylabel("what the synapse measures, log(Ca_teach / Ca_free) / k")
    a.set_title("D. What each synapse measures vs what it needs", loc="left", color=INK)
    a.legend(frameon=False, loc="upper left")
    fig.savefig(args.output / "ping_state.png", dpi=150)


if __name__ == "__main__":
    main()
