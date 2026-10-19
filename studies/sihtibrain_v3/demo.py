"""Reproduce V3 gate 1: neutral ping steering and soma observability controls."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sihtibrain_v3 import PATTERNS, cable, probe_echo, sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/gate1"))
    parser.add_argument("--trials", type=int, default=100,
                        help="Balanced noisy observations per seed; divisible by three")
    args = parser.parse_args()
    if args.trials < 3 or args.trials % 3:
        parser.error("--trials must be positive and divisible by three")
    args.output.mkdir(parents=True, exist_ok=True)

    adapted = sweep()
    frozen = sweep(beta=0)
    pair = adapted["conductance"][-2:]
    conditions = {
        "adaptive_two": (pair, True),
        "preset_two": (pair.copy(), True),  # same gates, external phase schedule
        "static_two": (np.repeat(pair[:1], 2, axis=0), True),
        "symmetric_two": (np.repeat([[.8, .8]], 2, axis=0), True),
        "linear_two": (pair, False),
    }
    labels = list(PATTERNS)
    patterns = [np.array(x) for x in PATTERNS.values()]
    prototypes = {key: np.array([[probe_echo(x, g, state_gate=on) for g in gates]
                                 for x in patterns]) for key, (gates, on) in conditions.items()}
    accuracy = {key: [] for key in conditions}
    for seed in (41, 97, 127, 211):
        rng = np.random.default_rng(seed)
        counts = {key: 0 for key in conditions}
        for k in range(args.trials):
            target = k % 3
            jitter = rng.normal(0., .05, 4)
            jitter -= jitter.mean()  # all classes remain soma matched on H0
            x = patterns[target] + jitter
            read_noise = rng.normal(0., .01, 2)
            for key, (gates, on) in conditions.items():
                echo = np.array([probe_echo(x, g, state_gate=on) for g in gates])
                observed = echo + read_noise
                predicted = int(np.argmin(np.linalg.norm(prototypes[key] - observed, axis=1)))
                counts[key] += int(target == predicted)
        for key in conditions:
            accuracy[key].append(counts[key] / args.trials)

    def contrast(run):
        flux = run["flux"]
        return np.abs(flux[:, 0] - flux[:, 1]) / flux.sum(axis=1)

    signatures = {key: {label: prototypes[key][i].tolist()
                        for i, label in enumerate(labels)} for key in conditions}
    data = {
        "protocol": {
            "tree": "7 compartment reciprocal leaky cable, roots 1 and 4, leaves 2 3 5 6",
            "ping": "identical somatic scalar on each equilibrium cycle",
            "adaptation": "a'=0.4 a + 0.6 normalized_branch_flux; g=0.8 exp(clip(-24(a-mean(a)), -1.6, 1.6))",
            "local_state_gate": "g_phase * (1 + 1.5 sigmoid(8 * (passive_root_voltage - 1.9)))",
            "matched_soma": "zero-sum leaf noise retains exact passive soma equality",
            "test_noise": {"leaf_current_std": .05, "echo_std": .01},
            "seeds": [41, 97, 127, 211], "trials_per_seed": args.trials,
        },
        "initial_passive_soma": {label: float(cable(x)[0][0]) for label, x in PATTERNS.items()},
        "adapted": {"late_flux_contrast_median": float(np.median(contrast(adapted)[20:])),
                    "late_alternation": float(np.mean(np.sign(np.diff(adapted["flux"], axis=1).ravel())[20:-1]
                                                        != np.sign(np.diff(adapted["flux"], axis=1).ravel())[21:])),
                    "last_gate_pair": pair.tolist()},
        "no_adaptation": {"late_flux_contrast_median": float(np.median(contrast(frozen)[20:]))},
        "signatures": signatures,
        "heldout_accuracy": accuracy,
    }
    (args.output / "metrics.json").write_text(json.dumps(data, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    for beta, color in [(0, "#888888"), (16, "#e28b38"), (24, "#2b6ea6")]:
        flux = sweep(beta=beta)["flux"]
        signed = (flux[:, 0] - flux[:, 1]) / flux.sum(axis=1)
        axes[0].plot(signed, label=f"adaptation gain {beta}", color=color)
    axes[0].set(title="A. Neutral ping steers after a transient", xlabel="probe cycle",
                ylabel="signed left-right flux contrast", ylim=(-.65, .65))
    axes[0].legend(fontsize=8)

    for label, point, color in zip(labels, prototypes["adaptive_two"],
                                    ("#246db2", "#d87a2c", "#288b68")):
        axes[1].scatter(*point, s=85, label=label, color=color)
    axes[1].set(title="B. Matched soma, two echoes", xlabel="phase 1 echo",
                ylabel="phase 2 echo")
    axes[1].legend(fontsize=8)
    axes[1].set_aspect("equal", adjustable="datalim")

    modes = list(conditions)
    mean = [np.mean(accuracy[key]) for key in modes]
    low = [min(accuracy[key]) for key in modes]
    high = [max(accuracy[key]) for key in modes]
    axes[2].bar(range(len(modes)), mean, color=["#2b6ea6", "#83aacb", "#e28b38", "#888888", "#b3b3b3"])
    axes[2].errorbar(range(len(modes)), mean,
                     yerr=[np.array(mean)-low, np.array(high)-mean], fmt="none", color="#222222", capsize=3)
    axes[2].set(title="C. Balanced held-out patterns", ylabel="nearest-prototype accuracy",
                ylim=(0, 1.05), xticks=range(len(modes)),
                xticklabels=[s.replace("_", "\n") for s in modes])
    axes[2].axhline(1/3, linestyle=":", color="#777777")
    fig.savefig(args.output / "gate1.svg")
    fig.savefig(args.output / "gate1.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"adapted": data["adapted"], "accuracy": accuracy}, indent=2))


if __name__ == "__main__":
    main()
