"""Executable controls for the preregistered Inhibitory Sensitivity Gate."""
import math
import unittest

from inhibitory_sensitivity_gate import (
    activation,
    exact_mechanism_checks,
    make_report,
    run_batch,
)


class InhibitorySensitivityTests(unittest.TestCase):
    def test_passive_is_exactly_matched(self):
        for u in (0.35, 0.50, 0.70, 0.85):
            self.assertAlmostEqual(activation(1-u), activation(1+u), places=14)

    def test_two_sides_have_opposite_inhibitory_response(self):
        checks = exact_mechanism_checks()
        self.assertLessEqual(checks["max_passive_mismatch"], 1e-12)
        self.assertTrue(checks["opposite_inhibitory_signs"])
        self.assertGreater(checks["min_absolute_inhibitory_change"], 0)

    def test_noise_free_probe_and_equal_cost_excitation(self):
        r = run_batch(seed=41, sigma=0.0)
        self.assertEqual(r["accuracy"]["inhibitory"], 1.0)
        self.assertEqual(r["accuracy"]["excitatory"], 1.0)
        self.assertEqual(r["accuracy"]["privileged_current_oracle"], 1.0)
        self.assertLess(r["accuracy"]["passive_only"], 0.60)
        self.assertLess(r["accuracy"]["sham_two_reads"], 0.60)

    def test_fixed_confirmatory_gate_and_failure_of_specificity(self):
        report = make_report()
        self.assertEqual(report["gates"]["G1_observable_regime"], "PASS")
        self.assertEqual(report["gates"]["G2_inhibition_specific"], "FAIL")
        self.assertEqual(report["gates"]["G3_matched_cost_adaptive_compute"], "NOT_TESTED")
        scores = report["conditions"]["0.015"]["mean_accuracy"]
        self.assertGreaterEqual(scores["inhibitory"], 0.95)
        self.assertLessEqual(scores["sham_two_reads"], 0.60)

    def test_cost_not_counted_as_free(self):
        c = make_report()["cost"]
        self.assertEqual(c["inhibitory"]["nonlinear_evaluations"], 2)
        self.assertEqual(c["excitatory"]["nonlinear_evaluations"], 2)
        self.assertEqual(c["sham_two_reads"]["nonlinear_evaluations"], 2)
        self.assertEqual(c["passive_only"]["nonlinear_evaluations"], 1)
        self.assertIn("NOT MEASURED", c["compute_savings"])

    def test_sham_is_noise_driven_but_scale_invariant(self):
        # A real sham comparison uses TWO independent noisy observations:
        # sign(sigma * (z_sham - z_baseline)) is unchanged when sigma > 0.
        # Reinitializing the seed reproduces the standard-normal draws.
        for seed in (41, 97, 127, 211):
            sham_at_scales = [
                run_batch(seed, sigma)["accuracy"]["sham_two_reads"]
                for sigma in (0.015, 0.04, 0.08)
            ]
            self.assertEqual(sham_at_scales[0], sham_at_scales[1])
            self.assertEqual(sham_at_scales[1], sham_at_scales[2])
            # With no noise both readings are exactly the same, and
            # the >0 tie rule gives class 0 for every balanced case.
            self.assertEqual(
                run_batch(seed, 0.0)["accuracy"]["sham_two_reads"], 0.5
            )

    def test_reproducible_and_validates_input(self):
        self.assertEqual(run_batch(97, 0.04), run_batch(97, 0.04))
        with self.assertRaises(ValueError):
            run_batch(41, -0.001)
        with self.assertRaises(ValueError):
            run_batch(41, 0.01, 3)


if __name__ == "__main__":
    unittest.main()
