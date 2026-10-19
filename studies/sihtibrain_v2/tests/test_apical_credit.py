"""Strict controls for Gate 4's apical-to-basal information route."""
import unittest
import numpy as np
from apical_credit import (Gate4, basal_features, basal_state, calcium,
                           nine_compartment_operator, tuft_to_soma, train)


class TuftCableTests(unittest.TestCase):
    def test_extended_cable_is_reciprocal_positive(self):
        h = nine_compartment_operator()
        np.testing.assert_allclose(h, h.T, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(h).min(), 0.)

    def test_tuft_to_soma_transfer_is_weak_and_disconnection_zeros_it(self):
        self.assertGreater(tuft_to_soma(), 0.)
        self.assertLess(tuft_to_soma(), .3)
        self.assertAlmostEqual(tuft_to_soma(trunk=0.), 0., 12)

    def test_symmetry_preserves_somatic_blindness_with_the_tuft_connected(self):
        v0 = basal_state([1,1,0,0])
        v1 = basal_state([1,0,1,0])
        self.assertAlmostEqual(v0[0], v1[0], 12)
        self.assertGreater(abs(v0[1]-v1[1]), .5)

    def test_nonlinear_branch_features_differentiate_cluster_and_split(self):
        self.assertGreater(basal_features([1,1,0,0]).sum(),
                           basal_features([1,0,1,0]).sum()+.7)


class LocalityTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1.,1.,0.,0.])
        self.m = Gate4.new(0)

    def test_context_only_changes_later_ping_not_basal_first_window(self):
        one = self.m.window(self.x, target=1.)
        zero = self.m.window(self.x, target=0.)
        self.assertAlmostEqual(one['predict'], zero['predict'], 12)
        np.testing.assert_allclose(one['eligibility'],zero['eligibility'],atol=1e-14)
        np.testing.assert_allclose(one['calcium_free'],zero['calcium_free'],atol=1e-14)
        self.assertGreater(one['ping_later'],zero['ping_later'])
        self.assertGreater(one['updates'].sum(),0.)
        self.assertLess(zero['updates'].sum(),0.)

    def test_local_update_never_needs_the_target(self):
        p = self.m.window(self.x,1.)
        np.testing.assert_allclose(p['updates'], p['eligibility'] *
                                   np.log(p['calcium_later']/p['calcium_free']))

    def test_without_context_or_ping_there_is_no_update(self):
        np.testing.assert_allclose(self.m.window(self.x,1.,context=False)['updates'],
                                   np.zeros(2),atol=1e-12)
        np.testing.assert_allclose(self.m.window(self.x,1.,ping=False)['updates'],
                                   np.zeros(2),atol=1e-12)

    def test_fractional_calcium_cancels_local_baseline_voltage(self):
        for attenuation in (.3,1.,2.):
            r0 = np.log(calcium(1.2,0.,1.,attenuation)/calcium(.9,0.,1.,attenuation))
            r1 = np.log(calcium(1.2,5.,1.,attenuation)/calcium(.9,5.,1.,attenuation))
            self.assertAlmostEqual(r0,r1,12)

    def test_expired_trace_blocks_basal_weight_write(self):
        np.testing.assert_allclose(self.m.window(self.x,1.,delay=30)['updates'],
                                   np.zeros(2),atol=1e-14)


class LearningControls(unittest.TestCase):
    def test_four_independent_seeds_solve_held_out_jittered_inputs(self):
        for seed in range(4):
            _,result,_ = train(seed,epochs=35)
            self.assertEqual(result['accuracy'],1.)
            self.assertGreater(result['margin'],.8)

    def test_no_ping_no_context_shuffled_target_no_basal_trace(self):
        for mode in ('no_ping','no_context','shuffled_context','no_basal_trace'):
            for seed in range(4):
                _,result,_=train(seed,mode=mode,epochs=35)
                self.assertLessEqual(result['accuracy'],2/3+1e-12,mode)

    def test_no_trunk_and_expired_trace(self):
        for kw in ({'trunk':0.},{'delay':25}):
            for seed in range(4):
                _,result,_=train(seed,epochs=35,**kw)
                self.assertLessEqual(result['accuracy'],2/3+1e-12)

    def test_linear_branch_readout_worse_than_nonlinear_features(self):
        accuracies=[train(s,epochs=35,linear=True)[1]['accuracy'] for s in range(4)]
        self.assertLess(max(accuracies),.9)

    def test_direct_gradient_baseline_also_solves_task(self):
        for seed in range(4):
            _,result,_=train(seed,mode='direct_gradient',epochs=35)
            self.assertEqual(result['accuracy'],1.)

    def test_weak_context_is_less_resilient_to_calcium_noise(self):
        strong=[train(s,epochs=35,noise=.03,context_gain=3.)[1]['accuracy']
                for s in range(4)]
        weak=[train(s,epochs=35,noise=.03,context_gain=.05)[1]['accuracy']
              for s in range(4)]
        self.assertEqual(strong,[1.]*4)
        self.assertLess(np.mean(weak),1.)


if __name__ == '__main__':
    unittest.main()
