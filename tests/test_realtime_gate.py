import unittest

import numpy as np

from tuft_gate import TuftTree, fresh, task, LEAVES
from realtime_gate import RealTime, updates

XS, YS = task()


class SignTheoremTests(unittest.TestCase):
    def test_single_cell_transfer_impedances_are_all_positive(self):
        """M-matrix: off-diagonals <= 0 and positive definite => inverse >= 0,
        so every active synapse's true gradient has the sign of the error."""
        for s in range(3):
            for trunk in (0, 8):
                t = fresh(s, trunk=trunk)
                for x in XS:
                    h = t.hessian(t.relax(x))
                    off = h - np.diag(np.diag(h))
                    self.assertLessEqual(off.max(), 0.0)
                    self.assertGreater(np.linalg.eigvalsh(h).min(), 0.0)
                    self.assertGreater(np.linalg.inv(h)[0, list(LEAVES)].min(), 0.0)


class RealTimeTests(unittest.TestCase):
    def setUp(self):
        self.tree = TuftTree(trunk=8, w=[0.7] * 4, b=-0.5)

    def mean_logratio(self, context, y, n=30, **kw):
        rt = RealTime(self.tree, **kw)
        rng = np.random.default_rng(3)
        vals = []
        for _ in range(n):
            rt.burst_theta = None
            dw, _ = updates(rt, XS[0], y, rng, "ping-local", context=context)
            vals.append(dw[:2].mean())
        return float(np.mean(vals)), float(np.std(vals) / np.sqrt(n))

    def test_warmup_removes_drift_without_context(self):
        m, se = self.mean_logratio(False, 0.6)
        self.assertLess(abs(m), 3 * se + 1e-3)

    def test_update_sign_follows_context_sign(self):
        up, _ = self.mean_logratio(True, 1.0)
        down, _ = self.mean_logratio(True, 0.0)
        self.assertGreater(up, 0.0)
        self.assertLess(down, 0.0)

    def test_bursts_follow_context(self):
        rt = RealTime(self.tree)
        rng = np.random.default_rng(1)
        diffs = {}
        for y in (1.0, 0.0):
            d = []
            for _ in range(20):
                rt.burst_theta = None
                _, _, be, bl, _ = rt.trial(XS[0], y, rng)
                d.append(bl - be)
            diffs[y] = np.mean(d)
        self.assertGreater(diffs[1.0], diffs[0.0])


if __name__ == "__main__":
    unittest.main()
