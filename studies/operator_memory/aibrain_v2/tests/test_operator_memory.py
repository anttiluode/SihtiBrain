"""Invariants and negative controls for AIbrainV2's operator memory."""
import unittest
import numpy as np
from operator_memory import (
    CUE_BITS,VALUE_BITS,MATRIX_BYTES,ConnectionMemory,IndexedMemory,
    encode_key,make_operator,response_matrix,euler_relaxation,dataset,
    DegreeNormalizedMemory,
    swapped_queries,run_case,benchmark,_score_by_class)


class OperatorTests(unittest.TestCase):
    def test_cable_symmetric_and_grounded_positive_definite(self):
        rng=np.random.default_rng(8)
        g=rng.random((16,8))*.2
        h=make_operator(g)
        np.testing.assert_allclose(h,h.T,atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(h).min(),0.)

    def test_first_order_degree_normalization_at_weak_coupling(self):
        rng=np.random.default_rng(27)
        w=rng.random((16,8))*1e-4
        baseline=DegreeNormalizedMemory()
        baseline.w[:]=w
        q=encode_key([1,3,5,7])
        np.testing.assert_allclose(baseline.read(q),response_matrix(w)@q,
                                   rtol=.005,atol=1e-11)

    def test_graph_schur_equals_direct_equilibrium_solver(self):
        rng=np.random.default_rng(7)
        g=rng.random((16,8))*.05
        q=encode_key([0,4,7,11])
        drive=np.r_[q,np.zeros(8)]
        direct=np.linalg.solve(make_operator(g),drive)[16:]
        schur=response_matrix(g)@q
        np.testing.assert_allclose(direct,schur,atol=1e-12)

    def test_euler_relaxation_converges_to_exact_equilibrium(self):
        rng=np.random.default_rng(6)
        g=rng.random((16,8))*.08
        key=encode_key([1,3,9,15])
        np.testing.assert_allclose(euler_relaxation(g,key)[16:],
                                   response_matrix(g)@key,atol=1e-7)

    def test_passive_gradient_energy_decreases(self):
        g=np.ones((16,8))*.04
        h=make_operator(g)
        q=np.r_[encode_key([0,1,2,3]),np.zeros(8)]
        z=np.zeros(24)
        vals=[]
        for _ in range(60):
            vals.append(.5*z@h@z-q@z)
            z+=.025*(q-h@z)
        self.assertLessEqual(float(np.max(np.diff(vals))),1e-11)

    def test_negative_conductance_rejected(self):
        w=np.zeros((16,8))
        w[0,0]=-.01
        with self.assertRaises(ValueError):
            make_operator(w)


class RetrievalTests(unittest.TestCase):
    def test_equal_storage_equal_writer(self):
        q=encode_key([1,5,8,12])
        graph=ConnectionMemory.empty(True)
        fast=ConnectionMemory.empty(False)
        for _ in range(4):
            graph.write(q,3)
            fast.write(q,3)
        self.assertEqual(graph.storage_bytes,512)
        self.assertEqual(fast.storage_bytes,MATRIX_BYTES)
        np.testing.assert_array_equal(graph.w,fast.w)
        self.assertEqual(int(graph.predict(q)),3)
        self.assertEqual(int(fast.predict(q)),3)

    def test_erase_connections_erases_recall_signal(self):
        g=ConnectionMemory.empty(True)
        x=encode_key([0,3,7,12])
        g.write(x,5)
        self.assertGreater(g.read(x)[5],0)
        g.erase_connections()
        np.testing.assert_allclose(g.read(x),np.zeros(8),atol=1e-14)

    def test_activity_reset_preserves_memory(self):
        g=ConnectionMemory.empty(True)
        x=encode_key([0,3,7,12])
        g.write(x,5)
        a=g.read(x).copy()
        g.reset_activity()
        np.testing.assert_allclose(g.read(x),a,atol=0.)

    def test_independent_noisy_queries_four_hot(self):
        x,_=dataset(4,16)
        corrupt=swapped_queries(x,300)
        self.assertEqual(corrupt.shape,(48,16))
        np.testing.assert_allclose(corrupt.sum(axis=1),4.)
        self.assertTrue(np.all(np.sum(np.abs(corrupt-x.repeat(3,axis=0)),axis=1)==2))

    def test_packed_index_footprint_and_rewrites(self):
        t=IndexedMemory()
        x=encode_key([0,2,6,10])
        t.write(x,3)
        self.assertEqual(t.storage_bytes,3)
        self.assertEqual(t.predict(x)[0],3)
        t.write(x,4)
        self.assertEqual(t.storage_bytes,3)
        self.assertEqual(t.predict(x)[0],4)

    def test_macro_metric_is_not_majority_accuracy(self):
        labels=np.array([0,0,0,0,1])
        pred=np.array([0,0,0,0,0])
        self.assertAlmostEqual(np.mean(labels==pred),.8)
        self.assertAlmostEqual(_score_by_class(pred,labels),.5)

    def test_same_operator_is_linear_in_read_cue(self):
        rng=np.random.default_rng(9)
        w=rng.random((16,8))*.06
        t=response_matrix(w)
        a=encode_key([0,2,4,6])
        b=encode_key([1,3,5,7])
        np.testing.assert_allclose(t@(a+b),t@a+t@b,atol=1e-13)


class ProtocolTests(unittest.TestCase):
    def test_reproducible_pairs_with_independent_labels(self):
        a,b=dataset(44,32)
        a2,b2=dataset(44,32)
        np.testing.assert_array_equal(a,a2)
        np.testing.assert_array_equal(b,b2)
        self.assertEqual(len({tuple(k) for k in a}),32)

    def test_per_case_writer_is_identical_even_after_rewrite(self):
        r=run_case(8,0)
        self.assertTrue(r['matrix_identical_after_rewrite'])
        self.assertTrue(all(r['activity_reset_invariant'].values()))
        self.assertEqual(r['stored_bytes']['operator'],512)
        self.assertEqual(r['stored_bytes']['fast_weight'],512)
        self.assertEqual(r['stored_bytes']['indexed'],24)

    def test_small_benchmark_has_reproducible_receipts(self):
        a=benchmark(loads=(8,),seeds=(0,1))
        b=benchmark(loads=(8,),seeds=(0,1))
        self.assertEqual(a['decision'],b['decision'])
        self.assertEqual(a['cases'],b['cases'])


if __name__=='__main__':
    unittest.main()
