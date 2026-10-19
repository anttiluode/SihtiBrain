"""Run the SihtiBrain thought experiment and save its traces and metrics."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sihtibrain import Circuit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/first_gate"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model = Circuit()
    cluster = np.array([0, 0, 1, 1, 0, 0, 0], dtype=float)
    split = np.array([0, 0, 1, 0, 0, 1, 0], dtype=float)
    passive_c = model.run(cluster)
    passive_s = model.run(split)
    active_c = model.run(cluster, coincidence_gain=0.5)
    active_s = model.run(split, coincidence_gain=0.5)
    pulse_c = model.run(cluster, soma_pulse=(120, 135, 3.0))
    pulse_s = model.run(split, soma_pulse=(120, 135, 3.0))
    times = 0.02 * np.arange(len(passive_c))
    threshold = float((active_c[-1, 0] + active_s[-1, 0]) / 2)
    energies = [model.energy(v, cluster) for v in passive_c]
    metrics = {
        "units": "dimensionless",
        "input": {"cluster_leaves": [2, 3], "split_leaves": [2, 5],
                  "current_per_active_leaf": 1.0},
        "passive_soma_max_abs_difference": float(np.max(abs(passive_c[:, 0] - passive_s[:, 0]))),
        "passive_branch_root_final_abs_difference": float(abs(passive_c[-1, 1] - passive_s[-1, 1])),
        "passive_axial_current_final_abs_difference": float(abs(
            model.axial_current(passive_c[-1], 1, 0)
            - model.axial_current(passive_s[-1], 1, 0))),
        "passive_energy_max_upward_step": float(max(np.diff(energies))),
        "passive_soma_ping_max_abs_difference": float(np.max(abs(pulse_c[:, 0] - pulse_s[:, 0]))),
        "passive_soma_ping_response_at_step_150": float(pulse_c[150, 0] - passive_c[150, 0]),
        "nonlinear_soma_cluster_final": float(active_c[-1, 0]),
        "nonlinear_soma_split_final": float(active_s[-1, 0]),
        "illustrative_threshold_calibrated_midpoint": threshold,
        "cluster_upward_crossings": model.threshold_crossings(active_c[:, 0], threshold),
        "split_upward_crossings": model.threshold_crossings(active_s[:, 0], threshold),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    positions = {0: (0, 0), 1: (-1.2, 1.1), 4: (1.2, 1.1),
                 2: (-1.8, 2.2), 3: (-0.6, 2.2),
                 5: (0.6, 2.2), 6: (1.8, 2.2)}
    ax = axes[0, 0]
    for i, j, _ in model.EDGES:
        ax.plot([positions[i][0], positions[j][0]],
                [positions[i][1], positions[j][1]], color="#64748b", lw=2, zorder=0)
    for i, (x, y) in positions.items():
        ax.scatter(x, y, s=500 if i == 0 else 340,
                   color="#16a6a2" if cluster[i] else "#233453", zorder=1)
        ax.text(x, y, str(i), ha="center", va="center", color="white", weight="bold")
    ax.text(0, 2.7, "Highlighted: cluster (2, 3). Split moves 3 to 5.",
            ha="center", fontsize=10)
    ax.set(xlim=(-2.3, 2.3), ylim=(-0.5, 3), title="Same input count and distance")
    ax.axis("off")

    ax = axes[0, 1]
    ax.plot(times, passive_c[:, 0], color="#374151", lw=3, label="passive cluster")
    ax.plot(times, passive_s[:, 0], color="#eab308", ls="--", lw=1.7, label="passive split")
    ax.plot(times, active_c[:, 0], color="#dc5661", lw=2, label="coincidence cluster")
    ax.plot(times, active_s[:, 0], color="#218aa1", ls=":", lw=2, label="coincidence split")
    ax.axhline(threshold, color="#dc5661", ls="--", alpha=.6, label="calibrated threshold")
    ax.set(title="Somatic voltage", xlabel="Time (model units)", ylabel="Voltage (model units)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(times, passive_c[:, 1], color="#dc5661", label="cluster branch 1")
    ax.plot(times, passive_s[:, 1], color="#218aa1", label="split branch 1")
    ax.plot(times, passive_c[:, 4], color="#dc5661", ls="--", label="cluster branch 2")
    ax.plot(times, passive_s[:, 4], color="#218aa1", ls="--", label="split branch 2")
    ax.set(title="Branch state preserves the distinction", xlabel="Time (model units)",
           ylabel="Voltage (model units)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(times, energies, color="#8b5cf6", lw=2)
    ax.set(title="Passive electrical energy, constant input", xlabel="Time (model units)",
           ylabel="E (model units)")

    fig.suptitle("SihtiBrain: local detail, somatic aliasing, branch coincidence", fontsize=14)
    fig.savefig(args.output / "sihtibrain.png", dpi=180)
    plt.close(fig)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
