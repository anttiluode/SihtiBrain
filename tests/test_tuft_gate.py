import unittest

import numpy as np

from tuft_gate import (task, fresh, train_exact, attenuation, alignment, Readout,
                       local_update, train_local, event_code, LEAVES)

XS, YS = task()


class CutVertexTests(unittest.TestCase):
    def test_apical_credit_factorises_through_the_soma_at_nonlinear_states(self):
        for trunk in (0, 3, 8):
            _, resid = attenuation(fresh(1, trunk=trunk), XS[0])
            self.assertLess(resid, 1e-12)

    def test_tuft_context_has_soma_direction_scaled_by_alpha(self):
        t = fresh(0, trunk=4)
        train_exact(t, XS, YS, 30)
        ro = Readout()
        for x, y in zip(XS, YS):
            ds, *_ = local_update(t, ro, x, y, 1e-3, "soma")
            dt, *_ = local_update(t, ro, x, y, 1e-3, "tuft")
            alpha, _ = attenuation(t, x)
            np.testing.assert_allclose(dt, alpha * ds, atol=2e-3 * np.abs(ds).max() + 1e-12)

    def test_attenuation_decays_with_trunk_length(self):
        a = [attenuation(fresh(0, trunk=L), XS[0])[0] for L in (0, 4, 8)]
        self.assertTrue(a[0] > a[1] > a[2] > 0)
        self.assertLess(a[2], 0.05)


class NoiseAndCodeTests(unittest.TestCase):
    def setUp(self):
        self.t = fresh(0, trunk=8)
        train_exact(self.t, XS, YS, 30)

    def test_noise_free_tuft_alignment_is_high_even_for_long_trunk(self):
        self.assertGreater(alignment(self.t, Readout(), XS, YS, 0.2, "tuft"), 0.95)

    def test_read_noise_destroys_passive_long_trunk_credit_but_not_ternary_event(self):
        rng = lambda: np.random.default_rng(10)
        passive = alignment(self.t, Readout(noise=0.01), XS, YS, 0.2, "tuft", rng())
        event = alignment(self.t, Readout(noise=0.01), XS, YS, 0.05, "ternary", rng(), dead=0.06)
        self.assertLess(passive, 0.5)
        self.assertGreater(event, 0.8)

    def test_event_codes(self):
        self.assertEqual(event_code(0.1, "ternary"), 1.0)
        self.assertEqual(event_code(-0.1, "ternary"), -1.0)
        self.assertEqual(event_code(0.01, "ternary", dead=0.06), 0.0)
        self.assertEqual(event_code(-0.5, "burst"), 0.0)


class TrainingTests(unittest.TestCase):
    def test_ternary_event_separates_and_one_sided_burst_does_not(self):
        t = fresh(0, trunk=8)
        _, outs, _ = train_local(t, Readout(noise=0.01), XS, YS, 0.05, "ternary",
                                 epochs=150, dead=0.06)
        self.assertGreater(outs[:2].min() - outs[2:].max(), 0.03)
        t = fresh(0, trunk=8)
        _, outs, _ = train_local(t, Readout(noise=0.01), XS, YS, 0.05, "burst",
                                 epochs=150, dead=0.06)
        self.assertLess(outs[:2].min() - outs[2:].max(), 0.0)


if __name__ == "__main__":
    unittest.main()
