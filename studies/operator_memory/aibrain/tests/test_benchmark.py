import unittest
from unittest.mock import patch
import numpy as np
from aibrain.benchmark import episode, corrupt, evaluate, run_benchmark, rewrite_targets, mechanism_gate
from aibrain.memory import ConnectionMemory


class BenchmarkTests(unittest.TestCase):
    def test_episode_is_balanced_reproducible_and_core_matched(self):
        x, y = episode(41, 32)
        x2, y2 = episode(41, 32)
        np.testing.assert_array_equal(x, x2)
        np.testing.assert_array_equal(y, y2)
        np.testing.assert_array_equal(x.sum(axis=1), np.full(32, 8))
        np.testing.assert_array_equal(np.bincount(y), np.full(8, 4))
        self.assertEqual(len(np.unique(x, axis=0)), 32)
        self.assertFalse(np.array_equal(x, episode(97, 32)[0]))

    def test_corruption_preserves_core_and_changes_exactly_requested_bits(self):
        x, _ = episode(41, 8)
        for swaps in (0, 1, 3):
            q = corrupt(x, swaps, np.random.default_rng(3))
            np.testing.assert_array_equal(q.sum(axis=1), x.sum(axis=1))
            np.testing.assert_array_equal((q != x).sum(axis=1), np.full(8, 2 * swaps))
        with self.assertRaises(ValueError):
            corrupt(x, 9, np.random.default_rng(3))

    def test_evaluation_reads_without_changing_connections(self):
        x, y = episode(41, 8)
        m = ConnectionMemory()
        for k, label in zip(x, y):
            m.write(k, np.eye(8)[label])
        before = m.weights.copy()
        scores = evaluate(m, x, y)
        self.assertGreaterEqual(scores, 0)
        self.assertLessEqual(scores, 1)
        np.testing.assert_array_equal(m.weights, before)

    def test_rewrite_truth_is_separate_from_old_truth(self):
        _, y = episode(41, 32)
        before = y.copy()
        updated, mask = rewrite_targets(y)
        np.testing.assert_array_equal(y, before)
        np.testing.assert_array_equal(updated[~mask], y[~mask])
        self.assertTrue(np.all(updated[mask] != y[mask]))
        self.assertEqual(mask.sum(), 8)

    def test_small_benchmark_has_controls_and_exact_matrix_tie(self):
        report = run_benchmark(seeds=[41], episodes=1, loads=[8])
        self.assertEqual(report['protocol']['seeds'], [41])
        rows = report['recall']
        methods = {r['method'] for r in rows}
        self.assertTrue({'connection', 'direct_equivalent', 'delta', 'hebbian',
                         'attention', 'pooled', 'erased', 'frozen', 'shuffled', 'raw_query'} <= methods)
        graph = [r for r in rows if r['method'] == 'connection']
        direct = [r for r in rows if r['method'] == 'direct_equivalent']
        self.assertEqual([r['accuracy'] for r in graph], [r['accuracy'] for r in direct])
        self.assertEqual(report['state_bytes']['connection'], 2048)
        self.assertEqual(report['state_bytes']['attention_by_load']['8'], 2304)
        for name, passed in report['mechanism']['passed'].items():
            self.assertTrue(passed, name)

    def test_mechanism_diagnostic_detects_a_read_that_corrupts_connections(self):
        original_read = ConnectionMemory.read
        def corrupting_read(memory, x, centered=True):
            result = original_read(memory, x, centered=centered)
            memory.weights[0, 0] += 1e-4
            return result
        with patch.object(ConnectionMemory, 'read', corrupting_read):
            report = mechanism_gate()
        self.assertFalse(report['passed']['queries_preserve_connections'])

    def test_invalid_protocol_is_rejected(self):
        for kwargs in ({'episodes': 0}, {'loads': [7]}, {'seeds': []}, {'loads': []}):
            with self.assertRaises(ValueError):
                run_benchmark(**kwargs)


if __name__ == '__main__':
    unittest.main()
