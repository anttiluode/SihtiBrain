"""Tests for sharp_address.py (CPU, seconds)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sharp_address as s  # noqa: E402
from vendor import tem_addressing as sol  # noqa: E402


class TestSharp(unittest.TestCase):
    def test_harness_reproduces_sol(self):
        for seed in (1000, 1001):
            for nv in (18, 30):
                mine = s.run_world(seed, nv, {"p": sol.DeltaMemory("phase")})
                his = sol.run_world(seed, nv)
                for ph in ("first_new_route", "rewritten", "untouched"):
                    self.assertEqual(mine[ph]["p"], his[ph]["phase_delta"])

    def test_expansion_shapes(self):
        g = sol.PhaseIntegrator().follow(("E", "N"))
        p = s.Expansion(64, 4)(g)
        self.assertEqual(int((p > 0).sum()), 4)
        self.assertAlmostEqual(float(np.linalg.norm(p)), 1.0)
        q = s.Expansion(64, 4, linear=True)(g)
        self.assertGreater(int((q != 0).sum()), 4)

    def test_linear_expansion_keeps_rank(self):
        E = s.Expansion(256, 16, linear=True)
        P = np.array([E(sol.PhaseIntegrator().follow(sol.canonical_path((x, y))))
                      for x in range(6) for y in range(6)])
        self.assertLessEqual(np.linalg.matrix_rank(P, tol=1e-8), sol.DIM)

    def test_expansion_is_route_invariant(self):
        E = s.Expansion(512, 16)
        for pos in [(0, 0), (2, 3), (5, 5)]:
            a = E(sol.PhaseIntegrator().follow(sol.canonical_path(pos)))
            b = E(sol.PhaseIntegrator().follow(sol.different_path(pos)))
            np.testing.assert_allclose(a, b, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
