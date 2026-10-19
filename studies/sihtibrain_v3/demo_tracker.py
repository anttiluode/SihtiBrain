"""V3 gate 2: causal multi-column sweep across noisy moving-bar movies."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sihtibrain_v3 import sweep
from tracker import local_evidence, make_movie, rollout, route, velocity_forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/tracker"))
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[41, 97, 127, 211, 303, 509, 701, 907] + list(range(1000, 1024)))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    adaptive = sweep()["flux"][-2:]
    controls = {
        "adaptive": adaptive,
        "preset": adaptive.copy(),
        "static": np.repeat(adaptive[:1], 2, axis=0),
        "symmetric": np.ones((2, 2)),
    }
    results = {key: [] for key in (*controls, "persistence", "velocity", "no_context")}
    example = None
    max_difference = 0.
    for seed in args.seeds:
        frames, truth = make_movie(seed)
        evidence = np.array([local_evidence(frame) for frame in frames])
        conditions = {key: rollout(frames, flux) for key, flux in controls.items()}
        stay = np.vstack([np.zeros(16), evidence[:-1]])
        move = stay.copy()
        for t in range(2, len(frames)):
            move[t] = velocity_forecast(evidence[t - 2], evidence[t - 1])
        conditions["persistence"] = {"prior": stay, "posterior": evidence}
        conditions["velocity"] = {"prior": move, "posterior": evidence}
        conditions["no_context"] = {"prior": np.zeros_like(evidence), "posterior": evidence}
        max_difference = max(max_difference, float(np.max(np.abs(
            conditions["adaptive"]["prior"] - conditions["symmetric"]["prior"]))))

        def score(run):
            top_prior = np.argsort(-run["prior"][1:], axis=1, kind="stable")[:, :3]
            top_post = np.argsort(-run["posterior"][1:], axis=1, kind="stable")[:, :3]
            recall = np.take_along_axis(truth[1:], top_prior, axis=1).sum(axis=1) / 3
            overlap = np.take_along_axis(truth[1:], top_post, axis=1).sum(axis=1) / 3
            lead_ranks = []
            for t in range(1, len(truth)):
                fresh = truth[t] & ~truth[t - 1]
                if not np.any(fresh):
                    continue
                candidates = run["prior"][t, ~truth[t - 1]]
                value = run["prior"][t, fresh][0]
                others = candidates[~fresh[~truth[t - 1]]]
                lead_ranks.append(float((np.sum(value > others) + .5 * np.sum(value == others)) / len(others)))
            return {"next_bar_top3_recall": float(np.mean(recall)),
                    "new_cell_rank_auc": float(np.mean(lead_ranks)),
                    "current_bar_top3_recall": float(np.mean(overlap))}

        for key, run in conditions.items():
            results[key].append(score(run))
        if example is None:
            example = (frames, truth, conditions)

    q = np.array([0., 0., 1., 1., 0., 0.])
    cancellation = float(np.max(np.abs(route(q, adaptive).mean(axis=0) -
                                        route(q, controls["symmetric"]).mean(axis=0))))
    paired = np.array([v["next_bar_top3_recall"] - a["next_bar_top3_recall"]
                       for a, v in zip(results["adaptive"], results["velocity"])])
    bootstrap = np.random.default_rng(2026).choice(
        paired, size=(10000, len(paired)), replace=True).mean(axis=1)
    data = {
        "protocol": {"columns": 16, "frames": 20, "bar_width": 3,
                     "seeds": args.seeds, "sensory_std": .35, "clutter_probability": .08,
                     "delays": "frame t column output reaches neighbors before frame t+1 sensory read",
                     "pulse_budget": "two equal somatic pulses per column per frame",
                     "receiver": "averages two pulse phases before next-frame update",
                     "no_oracle": "target masks used solely for evaluation"},
        "phase_average_vs_symmetric_max_abs": cancellation,
        "rollout_prior_vs_symmetric_max_abs": max_difference,
        "paired_comparison": {
            "velocity_minus_adaptive_paired_mean": float(np.mean(paired)),
            "positive_seeds": int(np.sum(paired > 0)),
            "ties": int(np.sum(paired == 0)),
            "bootstrap_95_percent": np.quantile(bootstrap, [.025, .975]).tolist(),
            "bootstrap_seed": 2026, "bootstrap_draws": 10000},
        "metrics": results,
        "means": {key: {metric: float(np.mean([row[metric] for row in values]))
                        for metric in values[0]} for key, values in results.items()},
    }
    (args.output / "metrics.json").write_text(json.dumps(data, indent=2) + "\n")

    frames, truth, conditions = example
    images = [(truth, "Actual moving bar"), (frames, "Noisy sensors"),
              (conditions["adaptive"]["prior"], "Adaptive, before new input"),
              (conditions["static"]["prior"], "Static direction, before input"),
              (conditions["symmetric"]["prior"], "Symmetric, before input"),
              (conditions["velocity"]["prior"], "Two-frame velocity baseline")]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), layout="constrained")
    for ax, (data_image, label) in zip(axes.flat, images):
        ax.imshow(data_image.T, interpolation="nearest", origin="lower", aspect="auto",
                  cmap="magma", vmin=0., vmax=1.)
        ax.set(title=label, xlabel="frame", ylabel="column", xlim=(0, 19))
    fig.suptitle("Gate 2: a two-phase ping loses its direction when the receiver averages it")
    fig.savefig(args.output / "tracker.svg")
    fig.savefig(args.output / "tracker.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"means": data["means"],
                      "rollout_prior_vs_symmetric_max_abs": max_difference}, indent=2))


if __name__ == "__main__":
    main()
