import unittest

import numpy as np

from sihtibrain_v3 import cable, probe_echo, sweep, PATTERNS


class CableTests(unittest.TestCase):
    def test_three_patterns_have_identical_passive_soma(self):
        values = [cable(x)[0][0] for x in PATTERNS.values()]
        np.testing.assert_allclose(values, values[0], rtol=0, atol=1e-12)

    def test_fixed_linear_operator_has_no_input_dependent_echo(self):
        g = (1.2, 0.5)
        echoes = [probe_echo(x, g, state_gate=False) for x in PATTERNS.values()]
        np.testing.assert_allclose(echoes, echoes[0], rtol=0, atol=1e-12)

    def test_bad_input_shape_rejected(self):
        with self.assertRaises(ValueError):
            cable([1, 0, 0])


class SweepTests(unittest.TestCase):
    def test_adaptation_yields_strong_alternation_after_transient(self):
        history = sweep(cycles=36, beta=24, retention=0.4, bias=1e-6)
        directions = np.sign(history["flux"][:, 0] - history["flux"][:, 1])
        self.assertGreater(np.mean(directions[16:-1] != directions[17:]), 0.9)
        contrasts = np.abs(np.diff(history["flux"], axis=1)).ravel() / history["flux"].sum(axis=1)
        self.assertGreater(np.median(contrasts[20:]), 0.45)
        self.assertLess(np.max(history["conductance"]), 4.0)
        self.assertGreater(np.min(history["conductance"]), 0.15)

    def test_no_adaptation_cannot_steer_a_useful_sweep(self):
        history = sweep(cycles=36, beta=0, retention=0.4, bias=1e-6)
        contrasts = np.abs(np.diff(history["flux"], axis=1)).ravel() / history["flux"].sum(axis=1)
        self.assertLess(np.max(contrasts), 1e-10)

    def test_exact_symmetry_stays_symmetric(self):
        history = sweep(cycles=8, beta=24, bias=0)
        np.testing.assert_allclose(history["flux"][:, 0], history["flux"][:, 1], atol=1e-12)

    def test_pulse_amplitude_does_not_set_direction(self):
        left = sweep(cycles=26, pulse=0.5)["flux"]
        right = sweep(cycles=26, pulse=2.0)["flux"]
        np.testing.assert_allclose(left * 4, right, rtol=1e-9)

    def test_invalid_parameters_rejected(self):
        for kwargs in ({"cycles": 0}, {"retention": 1.0}, {"pulse": 0}, {"beta": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sweep(**kwargs)


class ReadoutTests(unittest.TestCase):
    def test_two_neutral_sweep_phases_reveal_three_hidden_patterns(self):
        history = sweep(cycles=36)
        gates = history["conductance"][-2:]
        signatures = np.array([[probe_echo(x, g) for g in gates]
                               for x in PATTERNS.values()])
        separation = [np.linalg.norm(a - b) for i, a in enumerate(signatures)
                      for b in signatures[i + 1:]]
        self.assertGreater(min(separation), 1e-3)
        self.assertGreater(np.linalg.norm(gates[0] - gates[1]), 0.1)

    def test_without_local_state_gate_probe_is_blind_even_with_sweep(self):
        gates = sweep(cycles=36)["conductance"][-2:]
        signatures = np.array([[probe_echo(x, g, state_gate=False) for g in gates]
                               for x in PATTERNS.values()])
        np.testing.assert_allclose(signatures, np.tile(signatures[0], (3, 1)),
                                   rtol=0, atol=1e-12)

    def test_left_right_permutation_swaps_signatures_under_swapped_gate(self):
        g = (1.5, 0.6)
        left = probe_echo(PATTERNS["left"], g)
        right = probe_echo(PATTERNS["right"], g[::-1])
        self.assertAlmostEqual(left, right, places=12)


if __name__ == "__main__":
    unittest.main()
