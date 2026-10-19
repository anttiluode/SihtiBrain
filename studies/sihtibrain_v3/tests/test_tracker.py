import unittest

import numpy as np

from sihtibrain_v3 import sweep
from tracker import local_evidence, make_movie, route, rollout, velocity_forecast


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.phases = sweep()["flux"][-2:]

    def test_two_mirrored_phases_cancel_when_receiver_averages(self):
        signal = np.array([0., 1., 0., 0., 0.])
        two = route(signal, self.phases)
        self.assertGreater(abs(two[0, 0] - two[0, 2]), .4)
        self.assertLess((two[0, 0] - two[0, 2]) * (two[1, 0] - two[1, 2]), 0.)
        symmetric = route(signal, np.array([[1., 1.], [1., 1.]]))
        np.testing.assert_allclose(two.mean(axis=0), symmetric.mean(axis=0), atol=1e-12)

    def test_static_orientation_favors_one_neighbor(self):
        signal = np.array([0., 1., 0., 0., 0.])
        result = route(signal, np.repeat(self.phases[:1], 2, axis=0)).mean(axis=0)
        self.assertGreater(abs(result[0] - result[2]), .4)

    def test_movie_is_reproducible_and_mask_moves_one_column(self):
        a, truth = make_movie(41)
        b, truth_b = make_movie(41)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(truth, truth_b)
        self.assertEqual(a.shape, (20, 16))
        np.testing.assert_array_equal(truth.sum(axis=1), np.full(20, 3))
        center = np.argmax(truth, axis=1)
        np.testing.assert_array_equal(np.abs(np.diff(center)), np.ones(19))

    def test_no_future_frame_or_target_leaks_into_prediction(self):
        frames, _ = make_movie(97)
        first = rollout(frames, self.phases)
        changed = frames.copy()
        changed[7:] = 50.
        second = rollout(changed, self.phases)
        np.testing.assert_allclose(first["prior"][:7], second["prior"][:7])
        np.testing.assert_allclose(first["posterior"][:7], second["posterior"][:7])

    def test_preset_and_adaptive_two_phase_schedules_have_same_effect(self):
        frames, _ = make_movie(127)
        adaptive = rollout(frames, self.phases)
        preset = rollout(frames, self.phases.copy())
        np.testing.assert_array_equal(adaptive["prior"], preset["prior"])

    def test_velocity_forecast_uses_past_and_current_evidence_only(self):
        past = np.array([0., 1., 1., 0., 0., 0.])
        now = np.array([0., 0., 1., 1., 0., 0.])
        forecast = velocity_forecast(past, now)
        np.testing.assert_array_equal(forecast, [0., 0., 0., 1., 1., 0.])

    def test_invalid_signal_length_is_rejected(self):
        with self.assertRaises(ValueError):
            route(np.array([1.]), self.phases)
        with self.assertRaises(ValueError):
            local_evidence(np.array([1.]))


if __name__ == "__main__":
    unittest.main()
