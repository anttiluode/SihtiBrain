"""Gate 5 tests for matched inputs, local rules and batched implementation."""
import unittest
import numpy as np
from apical_credit import basal_features, Gate4
from gate5_stopping_test import (
    feature_map,make_data,metrics,updates,noise_draws,train_once,
    train_batched,one_seed,METHODS,LEARNING_RATES)


class Gate5Tests(unittest.TestCase):
    def test_vectorized_features_match_gate4(self):
        for x in ([1.,1.,0.,0.],[0.,1.,0.,1.],[.9,.8,.06,.05]):
            np.testing.assert_allclose(feature_map(np.asarray([x]))[0],
                                       basal_features(x),atol=1e-12)

    def test_identical_data_and_class_balance(self):
        p,y=make_data(101,24,'iid')
        self.assertEqual(p.shape,(144,2))
        self.assertEqual(int(y.sum()),48)
        np.testing.assert_array_equal(make_data(101,24,'iid')[0],p)
        self.assertFalse(np.array_equal(make_data(102,24,'iid')[0],p))

    def test_log_and_fraction_are_equal_for_small_nudge(self):
        early=np.ones(2)
        log=updates('two_ping_log',np.ones(2),1e-6,early,early)
        fractional=updates('three_factor_fractional',
                           np.ones(2),1e-6,early,early)
        np.testing.assert_allclose(log,fractional,rtol=1e-5,atol=1e-10)

    def test_noise_draws_are_paired(self):
        a=noise_draws(1,.03,3,9)
        np.testing.assert_array_equal(a,noise_draws(1,.03,3,9))
        self.assertFalse(np.array_equal(a,noise_draws(2,.03,3,9)))

    def test_event_and_no_context_do_not_use_calcium_noise(self):
        for mode in ('three_factor_event','three_factor_sign',
                     'no_context','direct_gradient'):
            actual=updates(mode,np.ones(2),.12,np.array([.1,.9]),
                           np.array([1.3,.3]))
            clean=updates(mode,np.ones(2),.12,np.ones(2),np.ones(2))
            np.testing.assert_array_equal(actual,clean)

    def test_no_context_does_not_update(self):
        data=make_data(1000,24,'iid')
        order=np.tile(np.arange(len(data[1])),(2,1))
        noise=np.ones((2,len(data[1]),2,2))
        w,b=train_once('no_context',0,.03,.2,data,order,noise,epochs=2)
        init=Gate4.new(0)
        np.testing.assert_array_equal(w,init.weights)
        self.assertEqual(b,init.bias)

    def test_balanced_accuracy_penalizes_majority_model(self):
        data=make_data(42,64,'iid')
        score=metrics(*data,np.zeros(2),-10.)
        self.assertAlmostEqual(score['accuracy'],2/3)
        self.assertAlmostEqual(score['balanced_accuracy'],.5)

    def test_batched_solver_agrees_with_independent_scalar_updates(self):
        data=make_data(1000,24,'iid')
        validation=make_data(2000,24,'iid')
        rng=np.random.default_rng(6000)
        order=np.array([rng.permutation(len(data[1])) for _ in range(3)])
        noise=noise_draws(0,.03,3,len(data[1]))
        packed=train_batched(0,data,validation,order,noise)
        for mode,(w,b,lr,loss) in packed.items():
            ref_w,ref_b=train_once(mode,0,.03,lr,data,order,noise,3)
            np.testing.assert_allclose(w,ref_w,atol=1e-12)
            self.assertAlmostEqual(b,ref_b,12)

    def test_validation_deterministic(self):
        a=one_seed(0,0.,epochs=2,lr_grid=(.02,.1))
        b=one_seed(0,0.,epochs=2,lr_grid=(.02,.1))
        for method in METHODS:
            self.assertEqual(a[method]['chosen_lr'],b[method]['chosen_lr'])
            self.assertEqual(a[method]['final_weights'],b[method]['final_weights'])


if __name__=='__main__':
    unittest.main()
