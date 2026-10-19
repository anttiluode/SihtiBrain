import unittest

import numpy as np

from ping_nudge import Tree, task, train
from ping_state import (PingReadout, ping_update, eqprop_update, alignment,
                        train_ping, fresh_tree)

XS, YS = task(hi=0.6, lo=0.3)
BASE = dict(ca_thr=5.0, ca_k=4.0)


def probe(seed=0):
    t = fresh_tree(seed)
    train(t, XS, YS, beta=0.02, lr=0.05, epochs=30, train_g=False, train_a=False, seed=seed)
    return t


class ExactnessTests(unittest.TestCase):
    def test_linear_tree_fractional_ping_rule_equals_eqprop_for_any_amplitude(self):
        t = Tree(a=0.0, w=[0.8, 0.5, 1.0, 0.7], b=-0.5)
        for amp in (0.5, 4.0):
            rd = PingReadout(amp=amp, **BASE)
            for x, y in zip(XS, YS):
                dw, _ = ping_update(t, rd, x, y, 1e-4)
                ew, _ = eqprop_update(t, x, y, 1e-4)
                np.testing.assert_allclose(dw, ew, atol=1e-3 * np.abs(ew).max())

    def test_absolute_rule_is_positive_rescaling_of_eqprop_below_threshold(self):
        t = fresh_tree(1)
        rd = PingReadout(**BASE)
        for x, y in zip(XS, YS):
            dw, _ = ping_update(t, rd, x, y, 1e-3, compare="diff")
            ew, _ = eqprop_update(t, x, y, 1e-3)
            m = np.abs(ew) > 1e-9
            self.assertTrue(np.all(dw[m] / ew[m] > 0))

    def test_bap_attenuation_cancels_in_the_fractional_rule(self):
        t = Tree(a=0.0, w=[0.8, 0.5, 1.0, 0.7], b=-0.5)
        a = PingReadout(**BASE)
        b = PingReadout(gamma=[0.2, 3.0, 1.0, 7.0], **BASE)
        for x, y in zip(XS, YS):
            np.testing.assert_allclose(ping_update(t, a, x, y, 1e-4)[0],
                                       ping_update(t, b, x, y, 1e-4)[0], atol=1e-3)


class SignLawTests(unittest.TestCase):
    def test_credit_aligns_below_and_inverts_past_the_unblock_midpoint(self):
        t = probe(0)
        self.assertGreater(alignment(t, PingReadout(**BASE), XS, YS), 0.95)
        self.assertLess(alignment(t, PingReadout(ca_thr=0.5, ca_k=4.0), XS, YS), -0.5)

    def test_no_ping_writes_nothing(self):
        dw, _ = ping_update(probe(0), PingReadout(**BASE), XS[0], YS[0], 0.02, use_ping=False)
        np.testing.assert_array_equal(dw, 0)


class LearningTests(unittest.TestCase):
    def test_two_pings_and_a_teacher_solve_the_task(self):
        t = fresh_tree(0)
        loss, outs, correct = train_ping(t, PingReadout(**BASE), XS, YS, epochs=200)
        self.assertEqual(correct, 6)
        self.assertGreater(outs[:2].min() - outs[2:].max(), 0.05)

    def test_past_midpoint_learns_the_inverted_task(self):
        t = fresh_tree(0)
        _, outs, correct = train_ping(t, PingReadout(ca_thr=0.5, ca_k=4.0), XS, YS, epochs=200)
        self.assertLess(outs[:2].min() - outs[2:].max(), 0)


if __name__ == "__main__":
    unittest.main()
