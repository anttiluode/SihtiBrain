import unittest

import numpy as np

from sihtibrain import Circuit


class CircuitTests(unittest.TestCase):
    def setUp(self):
        self.circuit = Circuit()
        self.cluster = np.array([0, 0, 1, 1, 0, 0, 0], dtype=float)
        self.split = np.array([0, 0, 1, 0, 0, 1, 0], dtype=float)

    def test_passive_relaxation_decreases_energy_for_constant_drive(self):
        trace = self.circuit.run(self.cluster, steps=1000)
        energies = np.array([self.circuit.energy(v, self.cluster) for v in trace])
        self.assertTrue(np.all(np.diff(energies) <= 1e-10))
        self.assertLess(energies[-1], energies[0])

    def test_soma_aliases_equal_depth_inputs_but_branches_retain_difference(self):
        clustered = self.circuit.run(self.cluster)
        split = self.circuit.run(self.split)
        np.testing.assert_allclose(clustered[:, 0], split[:, 0], atol=1e-12)
        self.assertGreater(abs(clustered[-1, 1] - split[-1, 1]), 0.1)
        self.assertGreater(abs(self.circuit.axial_current(clustered[-1], 1, 0)
                               - self.circuit.axial_current(split[-1], 1, 0)), 0.05)

    def test_branch_coincidence_separates_patterns_at_soma_and_spike_readout(self):
        passive = self.circuit.run(self.cluster)
        clustered = self.circuit.run(self.cluster, coincidence_gain=0.5)
        split = self.circuit.run(self.split, coincidence_gain=0.5)
        np.testing.assert_allclose(passive[:, 0], self.circuit.run(self.split)[:, 0], atol=1e-12)
        np.testing.assert_allclose(split[:, 0], passive[:, 0], atol=1e-12)
        self.assertGreater(clustered[-1, 0] - split[-1, 0], 0.2)
        threshold = (clustered[-1, 0] + split[-1, 0]) / 2
        self.assertEqual(self.circuit.threshold_crossings(clustered[:, 0], threshold), 1)
        self.assertEqual(self.circuit.threshold_crossings(split[:, 0], threshold), 0)

    def test_identical_somatic_ping_cannot_reveal_passive_hidden_branch_arrangement(self):
        pulse = (120, 135, 3.0)
        clustered = self.circuit.run(self.cluster, soma_pulse=pulse)
        split = self.circuit.run(self.split, soma_pulse=pulse)
        no_pulse = self.circuit.run(self.cluster)
        self.assertGreater(clustered[150, 0] - no_pulse[150, 0], 0.1)
        np.testing.assert_allclose(clustered[:, 0], split[:, 0], atol=1e-12)
        self.assertGreater(abs(clustered[-1, 1] - split[-1, 1]), 0.1)


if __name__ == '__main__':
    unittest.main()
