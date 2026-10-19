import unittest

import numpy as np

from ping_nudge import (Tree, task, pattern, linear_floor, eqprop_grad,
                        finite_difference_grad, implicit_grad,
                        soma_observability, sihti_residues, train, evaluate)


class ObservabilityTests(unittest.TestCase):
    def test_four_of_seven_modes_have_a_node_at_the_soma(self):
        _, phi, w0 = soma_observability(Tree())
        self.assertEqual(int(np.sum(np.abs(w0) < 1e-10)), 4)

    def test_cluster_minus_split_input_lives_only_in_soma_invisible_directions(self):
        lam, phi, w0 = soma_observability(Tree())
        diff = np.zeros(7); diff[3] = 1; diff[5] = -1      # {2,3} minus {2,5}
        coeff = phi.T @ diff
        visible = np.abs(w0) > 1e-10
        np.testing.assert_allclose(coeff[visible], 0, atol=1e-12)


class EqPropTests(unittest.TestCase):
    def setUp(self):
        self.tree = Tree(a=0.27, k=6, theta=1.1, w=[0.9, 0.7, 0.8, 0.6], b=-0.3)
        self.x, self.y = pattern((2, 3)), 0.6

    def test_implicit_gradient_matches_finite_differences(self):
        fd = finite_difference_grad(self.tree, self.x, self.y)
        ig = implicit_grad(self.tree, self.x, self.y)
        self.assertLess(np.linalg.norm(ig - fd) / np.linalg.norm(fd), 1e-6)

    def test_somatic_error_current_gives_exact_gradient_as_beta_shrinks(self):
        fd = finite_difference_grad(self.tree, self.x, self.y)
        errs = []
        for beta in (0.01, 0.001):
            ge, *_ = eqprop_grad(self.tree, self.x, self.y, beta)
            errs.append(np.linalg.norm(ge - fd) / np.linalg.norm(fd))
        self.assertLess(errs[1], 1e-4)
        self.assertGreater(errs[0] / errs[1], 50)      # second order in beta

    def test_synaptic_gradient_is_presynaptic_times_local_voltage_change(self):
        beta = 1e-3
        ge, _, vp, vm = eqprop_grad(self.tree, self.x, self.y, beta)
        local = -self.x * (vp[[2, 3, 5, 6]] - vm[[2, 3, 5, 6]]) / (2 * beta)
        np.testing.assert_allclose(ge[6:10], local, atol=1e-12)


class LearningTests(unittest.TestCase):
    def test_no_passive_tree_beats_the_affine_floor(self):
        xs, ys = task(hi=0.6, lo=0.3)
        floor, coef = linear_floor(xs, ys)
        self.assertAlmostEqual(floor, 0.01, places=12)
        rng = np.random.default_rng(3)
        for _ in range(5):
            t = Tree(a=0.0, g=rng.uniform(0.1, 2, 6), w=rng.normal(size=4), b=rng.normal())
            self.assertGreaterEqual(evaluate(t, xs, ys)[0], floor - 1e-12)

    def test_eqprop_trains_nonlinear_tree_past_the_floor(self):
        xs, ys = task(hi=0.6, lo=0.3)
        t = Tree(a=0.27, k=6, theta=1.1, w=[0.8, 0.6, 1.0, 0.5], b=-0.5)
        train(t, xs, ys, beta=0.02, lr=0.05, epochs=120, train_g=False, train_a=False)
        loss, outs, correct = evaluate(t, xs, ys)
        self.assertEqual(correct, 6)
        self.assertLess(loss, 0.5 * linear_floor(xs, ys)[0])


class KickTests(unittest.TestCase):
    def test_matched_soma_identical_ping_flips_only_the_clustered_branch(self):
        from scipy.optimize import brentq
        t = Tree(a=1.0, k=6, theta=2.0)
        xc = pattern((2, 3), 0.5)
        vc = t.relax(xc, steps=4000)
        amp = brentq(lambda A: t.relax(pattern((2, 5), A), steps=4000)[0] - vc[0], 0.3, 1.0, xtol=1e-12)
        xs = pattern((2, 5), amp)
        vs = t.relax(xs, steps=4000)
        self.assertAlmostEqual(vc[0], vs[0], places=9)
        tc, ec = t.trajectory(xc, 6000, v0=vc, pulse=(100, 150, 5.0))
        ts, es = t.trajectory(xs, 6000, v0=vs, pulse=(100, 150, 5.0))
        self.assertGreater(tc[-1, 0] - ts[-1, 0], 0.9)          # persistent plateau
        self.assertAlmostEqual(ts[-1, 0], vs[0], places=4)      # split returns
        self.assertLessEqual(np.diff(ec[150:]).max(), 1e-12)    # descent after the kick

    def test_linear_tree_ping_is_blind_by_superposition(self):
        t = Tree(a=0.0)
        a, _ = t.trajectory(pattern((2, 3), 0.5), 800, pulse=(100, 150, 5.0))
        b, _ = t.trajectory(pattern((2, 5), 0.5), 800, pulse=(100, 150, 5.0))
        np.testing.assert_allclose(a[:, 0], b[:, 0], atol=1e-12)


class SihtiIdentityTests(unittest.TestCase):
    def test_diffusion_residue_is_axial_plus_leak_current_and_sums_back(self):
        t = Tree()
        h = t.laplacian() + t.leak * np.eye(7)
        x0 = np.random.default_rng(0).normal(size=7)
        res, path = sihti_residues(h, x0, 0.05, 50)
        np.testing.assert_allclose(res.sum(0) + path[-1], x0, atol=1e-14)
        k = 7
        current_out = np.array([sum(g * (path[k][i] - path[k][j]) for (a_, b_, _), g in zip(
            __import__("ping_nudge").EDGES, t.g) for i, j in ((a_, b_), (b_, a_)) if i == n)
            for n in range(7)]) + t.leak * path[k]
        np.testing.assert_allclose(res[k], 0.05 * current_out, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
