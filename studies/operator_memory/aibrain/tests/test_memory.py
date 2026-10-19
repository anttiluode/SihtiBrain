import unittest
import numpy as np

from aibrain.memory import ConnectionMemory, split, cue
from aibrain.baselines import HebbianMemory, DeltaMemory, HistoryAttention


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.keys = np.eye(16, dtype=np.float32).reshape(4, 4, 16).sum(axis=1)
        self.values = np.eye(4, dtype=np.float32)

    def test_split_reconstructs_and_equal_cores_hide_identity(self):
        for x in self.keys:
            core, residue = split(x)
            np.testing.assert_allclose(core + residue, x)
            self.assertEqual(core, 0.25)
        self.assertFalse(np.array_equal(split(self.keys[0])[1], split(self.keys[1])[1]))

    def test_online_association_survives_activity_reset(self):
        m = ConnectionMemory(16, 4)
        for x, v in zip(self.keys, self.values):
            m.write(x, v)
        before = m.weights.copy()
        for i, x in enumerate(self.keys):
            self.assertEqual(int(np.argmax(m.read(x))), i)
            m.clear_activity()
            self.assertTrue(np.all(m.activity == 0))
            self.assertEqual(int(np.argmax(m.read(x))), i)
        np.testing.assert_array_equal(m.weights, before)
        self.assertEqual(m.state_bytes, 16 * 4 * 4)
        m.erase()
        np.testing.assert_array_equal(m.read(self.keys[0]), np.zeros(4))

    def test_local_rewrite_changes_binding_and_keeps_disjoint_binding(self):
        m = ConnectionMemory(16, 4)
        for x, v in zip(self.keys, self.values):
            m.write(x, v)
        for _ in range(4):
            m.write(self.keys[0], self.values[2])
        self.assertEqual(np.argmax(m.read(self.keys[0])), 2)
        self.assertEqual(np.argmax(m.read(self.keys[1])), 1)
        self.assertGreaterEqual(m.weights.min(), 0)
        self.assertLessEqual(m.weights.max(), 1)

    def test_relaxation_descends_energy_to_direct_equilibrium(self):
        m = ConnectionMemory(16, 4)
        for x, v in zip(self.keys, self.values):
            m.write(x, v)
        expected = m.read(self.keys[1])
        h, energies = m.relax(self.keys[1], steps=80)
        np.testing.assert_allclose(h, expected, atol=1e-9)
        self.assertTrue(np.all(np.diff(energies) <= 1e-12))
        q = cue(self.keys[1])
        np.testing.assert_allclose(expected, m.weights.T @ q / (1 + m.weights.sum(axis=0)), atol=1e-9)

    def test_constant_query_is_explicitly_uninformative(self):
        for cls in (ConnectionMemory, HebbianMemory, DeltaMemory, HistoryAttention):
            m = cls(16, 4)
            m.write(self.keys[0], self.values[0])
            np.testing.assert_array_equal(m.read(np.ones(16)), np.zeros(4))
            np.testing.assert_array_equal(m.read(np.zeros(16)), np.zeros(4))

    def test_all_baselines_recall_orthogonal_associations(self):
        for cls in (HebbianMemory, DeltaMemory, HistoryAttention):
            m = cls(16, 4)
            for x, v in zip(self.keys, self.values):
                m.write(x, v)
            for i, x in enumerate(self.keys):
                self.assertEqual(np.argmax(m.read(x)), i, cls.__name__)
            expected = 4 * (16 + 4) * 4 if cls is HistoryAttention else 16 * 4 * 4
            self.assertEqual(m.state_bytes, expected)

    def test_large_finite_cues_remain_finite_and_constant_cues_are_zero(self):
        for cls in (ConnectionMemory, HebbianMemory, DeltaMemory, HistoryAttention):
            m = cls(16, 4)
            m.write(self.keys[0], self.values[0])
            np.testing.assert_array_equal(m.read(np.full(16, 1e308)), np.zeros(4))
            self.assertTrue(np.isfinite(m.read(np.array([1e308, -1e308] * 8))).all())
        core, residue = split(np.full(16, 1e308))
        self.assertEqual(core, 1e308)
        np.testing.assert_array_equal(residue, np.zeros(16))

    def test_probability_tolerance_cannot_create_unbounded_conductance(self):
        m = ConnectionMemory(16, 4)
        with self.assertRaises(ValueError):
            m.write(self.keys[0], [1.000009, 0, 0, 0])
        for _ in range(64):
            m.write(self.keys[0], [1.000000009, 0, 0, 0])
        self.assertLessEqual(m.weights.max(), 1)

    def test_bad_input_is_rejected_before_memory_changes(self):
        for cls in (ConnectionMemory, HebbianMemory, DeltaMemory, HistoryAttention):
            with self.assertRaises(ValueError):
                cls(0, 4)
            m = cls(16, 4)
            for x, v in [(np.ones(15), self.values[0]),
                         (np.full(16, np.nan), self.values[0]),
                         (np.ones(16) * 2, self.values[0]),
                         (self.keys[0], np.ones(4)),
                         (self.keys[0], np.array([np.inf, 0, 0, 0]))]:
                with self.assertRaises(ValueError):
                    m.write(x, v)
            with self.assertRaises(ValueError):
                m.read(np.full(16, np.inf))
        for kwargs in ({'eta': -1}, {'eta': 2}, {'leak': 0}, {'leak': np.nan}):
            with self.assertRaises(ValueError):
                ConnectionMemory(16, 4, **kwargs)


if __name__ == '__main__':
    unittest.main()
