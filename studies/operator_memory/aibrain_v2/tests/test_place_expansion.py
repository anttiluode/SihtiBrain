"""Gate 4 structural, leakage, budget and sensory-noise invariants."""
import unittest
import numpy as np
from learned_relations import learned_sources,spectral_codes,graph_state
from tem_addressing import make_world,canonical_path,different_path
from place_expansion import (
    NSTATES,DIM,codebooks,sparse_relu,IndexedGraph,KeyMemory,
    path_state,visited_states,corrupt_walk,belief_filter,
    trained_memories,noisy_trial,run_world,benchmark)

class ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table,cls.agree=learned_sources()
        cls.spectral,_=spectral_codes(cls.table)
        cls.books=codebooks(cls.spectral)

    def test_source_graph_invariant_and_no_label_dependent_codes(self):
        self.assertTrue(self.agree)
        a=codebooks(self.spectral)
        for k in a:
            np.testing.assert_array_equal(a[k][0],self.books[k][0])

    def test_sparse_and_dense_counts(self):
        self.assertEqual(self.books['sparse64'][0].shape,(36,64))
        self.assertEqual(self.books['sparse256'][0].shape,(36,256))
        self.assertTrue(np.all((self.books['sparse256'][0]>0).sum(axis=1)<=16))
        self.assertTrue(np.all((self.books['sparse64'][0]>0).sum(axis=1)<=8))
        self.assertGreater((self.books['dense256'][0]>0).sum(),36*16)

    def test_nonnegative_normalized_sparse_features(self):
        for name in ('sparse64','sparse256','dense256','random_state256'):
            matrix=self.books[name][0]
            self.assertTrue(np.all(matrix>=0.))
            np.testing.assert_allclose(np.linalg.norm(matrix,axis=1),
                                       np.ones(36),atol=2e-7)

    def test_projected_codes_are_state_distinguishable(self):
        for name in ('spectral12','sparse64','sparse256','onehot36'):
            codes=self.books[name][0]
            d=np.linalg.norm(codes[:,None,:]-codes[None,:,:],axis=2)
            np.fill_diagonal(d,100.)
            self.assertGreater(float(d.min()),1e-6)

    def test_full_state_onehot_has_no_aliasing(self):
        code=self.books['onehot36'][0]
        self.assertEqual(code.shape,(NSTATES,NSTATES))
        np.testing.assert_allclose(code@code.T,np.eye(NSTATES),atol=1e-14)

    def test_budget_counts_both_projection_and_fast_weights(self):
        models,index=trained_memories(
            self.table,self.spectral,*make_world(1000)[:3],np.arange(18))
        # Use the explicit budget formula rather than trusting a label.
        self.assertEqual(models['spectral12'].storage_bytes,2256)
        self.assertEqual(models['sparse256'].storage_bytes,
                         144+1728+12288+8192)
        self.assertEqual(models['sparse64'].storage_bytes,
                         144+1728+64*12*4+8*64*4)
        self.assertEqual(models['random_state256'].storage_bytes,
                         144+36*256*4+8192)
        self.assertEqual(models['onehot36'].storage_bytes,144+8*36*4)
        self.assertEqual(index.storage_bytes,144+18)

    def test_same_state_by_two_paths(self):
        for pos in ((0,0),(2,3),(5,5)):
            self.assertEqual(path_state(self.table,canonical_path(pos)),
                             path_state(self.table,different_path(pos)))

    def test_delta_updates_rewrite_value(self):
        books=self.books
        k=books['onehot36'][0]
        model=KeyMemory(k,144,0,0)
        state=17
        for _ in range(3):model.write_state(state,2)
        self.assertEqual(model.predict_state(state),2)
        for _ in range(3):model.write_state(state,6)
        self.assertEqual(model.predict_state(state),6)

    def test_fixed_index_rewrite(self):
        index=IndexedGraph(144)
        index.write_state(20,3)
        index.write_state(15,5)
        index.write_state(20,6)
        self.assertEqual(index.predict_state(15),5)
        self.assertEqual(index.predict_state(20),6)
        self.assertEqual(index.storage_bytes,146)


class SensoryCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table,_=learned_sources()
        cls.spectral,_=spectral_codes(cls.table)

    def test_no_corruption_gives_same_action_and_belief(self):
        world=make_world(1000)
        places,symbols,labels,order=world
        known=visited_states(self.table,places,order[:18],symbols)
        route=different_path((3,4))
        obs,pos=corrupt_walk(route,symbols,np.random.default_rng(0),0.,0.)
        self.assertEqual(tuple(a for a,_ in obs),route)
        self.assertEqual(pos,(3,4))
        posterior=belief_filter(self.table,obs,known,0.,0.)
        target=path_state(self.table,route)
        self.assertAlmostEqual(float(posterior[target]),1.,12)

    def test_sensor_map_contains_only_visited_nodes(self):
        places,symbols,labels,order=make_world(1000)
        v=order[:18]
        known=visited_states(self.table,places,v,symbols)
        self.assertEqual(int(np.sum(known>=0)),18)
        self.assertEqual(set(known[known>=0]),set(symbols[v]))

    def test_bayes_filter_normalized_and_not_oracle(self):
        places,symbols,values,order=make_world(1001)
        known=visited_states(self.table,places,order[:18],symbols)
        obs,pos=corrupt_walk(different_path((4,5)),symbols,
                             np.random.default_rng(44),.15)
        posterior=belief_filter(self.table,obs,known,.15)
        self.assertAlmostEqual(float(posterior.sum()),1.,12)
        self.assertTrue(np.all(posterior>=0.))
        self.assertEqual(posterior.shape,(36,))

    def test_all_sensors_unknown_has_uninformative_likelihood(self):
        route=different_path((3,3))
        symbols=np.zeros(36,dtype=int)
        obs,_=corrupt_walk(route,symbols,np.random.default_rng(9),.15,0.)
        p=belief_filter(self.table,obs,np.full(36,-1,dtype=int),.15,0.)
        self.assertAlmostEqual(p.sum(),1.,12)
        self.assertLess(p.max(),1.)

    def test_oracle_true_path_never_misses_seen_binding(self):
        places,symbols,values,order=make_world(1000)
        visited=order[:18]
        models,index=trained_memories(self.table,self.spectral,places,
                                      symbols,values,visited)
        known=visited_states(self.table,places,visited,symbols)
        report=noisy_trial(self.table,places,symbols,values,visited,
                           known,models,index,1000,18,.30)
        self.assertAlmostEqual(report['oracle_index'],1.)
        self.assertTrue(0.<=report['true_state_probability']<=1.)
        self.assertTrue(0.<=report['map_index']<=1.)

    def test_quick_run_is_deterministic(self):
        a=benchmark(seeds=(1000,1001))
        b=benchmark(seeds=(1000,1001))
        self.assertEqual(a['decisions'],b['decisions'])
        self.assertEqual(a['cases'],b['cases'])
        self.assertEqual(a['summary']['18']['bytes']['sparse256'],22352.)

if __name__=='__main__':unittest.main()
