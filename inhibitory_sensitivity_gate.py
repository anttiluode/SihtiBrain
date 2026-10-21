"""Inhibitory Sensitivity Gate: a locked, non-biophysical observability toy.

Preregistration: docs/INHIBITORY_SENSITIVITY_PREREG.md.
Standard library only; no fitted classifier and no adaptive computation claim.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

SEEDS = (41, 97, 127, 211)
SIGMAS = (0.015, 0.04, 0.08)
TRIALS_PER_SEED = 800
EPSILON = 0.10
LOW_U = 0.35
HIGH_U = 0.85


def activation(current: float) -> float:
    """Smooth, peaked *illustration*, not a calcium-spike model."""
    return math.exp(-((current - 1.0) ** 2))


def exact_mechanism_checks() -> dict:
    """Check equal passive outputs and opposite inhibitory response signs."""
    max_passive_mismatch = 0.0
    sign_ok = True
    smallest_inhibitory_change = float("inf")
    for k in range(51):
        u = LOW_U + (HIGH_U - LOW_U) * k / 50
        weak, over = 1.0 - u, 1.0 + u
        max_passive_mismatch = max(
            max_passive_mismatch, abs(activation(weak) - activation(over))
        )
        weak_change = activation(weak - EPSILON) - activation(weak)
        over_change = activation(over - EPSILON) - activation(over)
        sign_ok &= weak_change < 0.0 and over_change > 0.0
        smallest_inhibitory_change = min(
            smallest_inhibitory_change, -weak_change, over_change
        )
    return {
        "max_passive_mismatch": max_passive_mismatch,
        "opposite_inhibitory_signs": bool(sign_ok),
        "min_absolute_inhibitory_change": smallest_inhibitory_change,
    }


def run_batch(seed: int, sigma: float, trials: int = TRIALS_PER_SEED) -> dict:
    if trials <= 0 or trials % 2:
        raise ValueError("trials must be a positive even integer")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")

    rng = random.Random(seed)
    # Pair weak and overdriven cases at EXACTLY the same u.
    # No information about label or u is given to any non-oracle reader.
    cases = []
    for _ in range(trials // 2):
        u = rng.uniform(LOW_U, HIGH_U)
        cases.extend(((0, u), (1, u)))
    rng.shuffle(cases)

    correct = {
        "inhibitory": 0,
        "excitatory": 0,
        "sham_two_reads": 0,
        "passive_only": 0,
        "privileged_current_oracle": 0,
    }
    change_sums = {"weak_inhibitory": 0.0, "over_inhibitory": 0.0}
    for label, u in cases:
        current = 1.0 + (u if label else -u)
        # Shared noisy first read across all arms, so no probe benefits
        # from a more favorable first observation.
        baseline = activation(current) + rng.gauss(0.0, sigma)
        inhibited = activation(current - EPSILON) + rng.gauss(0.0, sigma)
        excited = activation(current + EPSILON) + rng.gauss(0.0, sigma)
        sham = activation(current) + rng.gauss(0.0, sigma)
        delta_inh = inhibited - baseline
        delta_exc = excited - baseline
        delta_sham = sham - baseline

        # Frozen direction-of-change decoders; never fit on test cases.
        predictions = {
            "inhibitory": int(delta_inh > 0.0),
            "excitatory": int(delta_exc < 0.0),
            "sham_two_reads": int(delta_sham > 0.0),
            "passive_only": int(baseline > 0.75),
            # Intentionally forbidden to other readers.
            "privileged_current_oracle": int(current > 1.0),
        }
        for name, prediction in predictions.items():
            correct[name] += int(prediction == label)
        change_sums["over_inhibitory" if label else "weak_inhibitory"] += delta_inh

    return {
        "seed": seed,
        "sigma": sigma,
        "trials": trials,
        "accuracy": {key: count / trials for key, count in correct.items()},
        "mean_inhibitory_delta": {
            key: value / (trials // 2) for key, value in change_sums.items()
        },
    }


def make_report() -> dict:
    checks = exact_mechanism_checks()
    conditions = {}
    for sigma in SIGMAS:
        batches = [run_batch(seed, sigma) for seed in SEEDS]
        scores = {
            name: sum(batch["accuracy"][name] for batch in batches) / len(batches)
            for name in batches[0]["accuracy"]
        }
        conditions[str(sigma)] = {"mean_accuracy": scores, "per_seed": batches}
    primary = conditions[str(SIGMAS[0])]["mean_accuracy"]
    mechanism_pass = (
        checks["max_passive_mismatch"] <= 1e-12
        and checks["opposite_inhibitory_signs"]
        and primary["inhibitory"] >= 0.95
        and primary["sham_two_reads"] <= 0.60
        and primary["inhibitory"] - primary["sham_two_reads"] >= 0.30
    )
    inhibition_specific_pass = (
        primary["inhibitory"] - primary["excitatory"] >= 0.05
    )
    return {
        "protocol": "docs/INHIBITORY_SENSITIVITY_PREREG.md",
        "model": "A(I)=exp(-(I-1)^2); illustrative, NOT biophysical",
        "parameters": {
            "seeds": list(SEEDS), "sigmas": list(SIGMAS),
            "trials_per_seed": TRIALS_PER_SEED, "epsilon": EPSILON,
            "distance_from_peak": [LOW_U, HIGH_U],
        },
        "analytical_checks": checks,
        "conditions": conditions,
        "cost": {
            "passive_only": {"nonlinear_evaluations": 1, "scalar_reads": 1},
            "inhibitory": {"nonlinear_evaluations": 2, "scalar_reads": 2},
            "excitatory": {"nonlinear_evaluations": 2, "scalar_reads": 2},
            "sham_two_reads": {"nonlinear_evaluations": 2, "scalar_reads": 2},
            "privileged_current_oracle": "UNFAIR: reads hidden current",
            "compute_savings": "NOT MEASURED; none may be inferred",
        },
        "gates": {
            "G1_observable_regime": "PASS" if mechanism_pass else "FAIL",
            "G2_inhibition_specific": (
                "PASS" if inhibition_specific_pass else "FAIL"
            ),
            "G3_matched_cost_adaptive_compute": "NOT_TESTED",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = make_report()
    data = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="utf-8")
    print("Primary accuracies:", report["conditions"]["0.015"]["mean_accuracy"])
    print("Gates:", report["gates"])
    if args.output is not None:
        print("Receipt:", args.output)


if __name__ == "__main__":
    main()
