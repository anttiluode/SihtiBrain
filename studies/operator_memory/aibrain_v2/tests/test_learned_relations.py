"""Gate 3 invariants: learned transitions, transfer and controlled memory sizes."""
import unittest
import numpy as np
from learned_relations import (
    SOURCE_SEEDS,WALK_STEPS,ACTIONS,NSTATES,
    source_observations,learn_automaton,learned_sources,
    graph_state,spectral_codes,random_codes,path_invariants,
    bad_action_control,LearnedMatrix,LearnedIndexed,
    run_target,benchmark)
from tem_addressing import canonical_path,different_path

class StructuralLearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table,cls.agree=learned_sources()
        cls.spectral,_=spectral_codes(cls.table)
        cls.random=random_codes(42)

    def test_training_reads_opaque_strings_not_coordinates(self):
        anchor,observations=source_observations(3000)
        self.assertIsInstance(anchor,str)
        self.assertEqual(len(observations),WALK_STEPS)
        self.assertEqual(len(observations[0]),3)
        self.assertTrue(all(isinstance(x,str) for x in observations[0]))
        self.assertEqual(learn_automaton(anchor,observations).shape,(36,4))

    def test_all_eight_worlds_recover_same_structure(self):
        self.assertTrue(self.agree)
        self.assertEqual(self.table.shape,(NSTATES,len(ACTIONS)))
        for a in range(len(ACTIONS)):
            self.assertEqual(len(set(self.table[:,a])),36)

    def test_same_state_after_new_route(self):
        for x in range(6):
            for y in range(6):
                p=(x,y)
                self.assertEqual(graph_state(self.table,canonical_path(p)),
                                 graph_state(self.table,different_path(p)))

    def test_closed_loop_and_disjoint_structural_addresses(self):
        v=path_invariants(self.table,self.spectral)
        self.assertEqual(v['unique_states'],36)
        self.assertEqual(v['route_errors'],0)
        self.assertTrue(v['closed_loop_returns'])
        self.assertGreater(v['minimum_code_distance'],1e-6)

    def test_shuffled_action_control_rejected(self):
        self.assertTrue(bad_action_control())

    def test_missing_transition_rejected_without_oracle_fill(self):
        anchor,obs=source_observations(3000)
        with self.assertRaises(ValueError):
            learn_automaton(anchor,obs[:100])

    def test_conflicting_transition_rejected(self):
        anchor,obs=source_observations(3000)
        first=obs[0]
        other=next(x[2] for x in obs if x[2]!=first[2])
        with self.assertRaises(ValueError):
            learn_automaton(anchor,obs+[(first[0],first[1],other)])

    def test_memory_storage_counts_structure_and_code(self):
        m=LearnedMatrix(self.table,self.spectral)
        self.assertEqual(m.w.nbytes,384)
        self.assertEqual(m.codes.nbytes,1728)
        self.assertEqual(m.table.nbytes,144)
        self.assertEqual(m.byte_count,2256)

    def test_same_learning_rule_random_vs_spectral(self):
        for codes in (self.spectral,self.random):
            m=LearnedMatrix(self.table,codes)
            p=canonical_path((1,3))
            q=different_path((1,3))
            for _ in range(3):m.write(p,1,5)
            self.assertEqual(m.predict(q,0),5)

    def test_classical_index_writes_new_label_and_old_key(self):
        m=LearnedIndexed(self.table)
        a=canonical_path((3,4));b=different_path((3,4))
        m.write(a,0,2)
        self.assertEqual(m.predict(b,3),2)
        m.write(b,3,7)
        self.assertEqual(m.predict(a,0),7)
        self.assertEqual(m.byte_count,145)

    def test_reproducible_fresh_worlds_with_same_inputs(self):
        a=run_target(1000,18,self.table,self.spectral,self.random)
        b=run_target(1000,18,self.table,self.spectral,self.random)
        self.assertEqual(a,b)
        self.assertEqual(a['bytes']['learned_spectral_delta'],2256)
        self.assertEqual(a['bytes']['learned_indexed'],162)
        self.assertEqual(a['new_route']['learned_indexed'],1.)
        self.assertEqual(a['rewritten']['learned_indexed'],1.)
        self.assertEqual(a['untouched']['learned_indexed'],1.)

    def test_two_world_smoke_passes_necessary_gate(self):
        res=benchmark(source_seeds=(3000,3001),target_seeds=(1000,1001))
        self.assertTrue(res['decisions']['gate0_structure_invariance'])
        self.assertEqual(res['summary']['18']['learned_indexed']
                         ['new_route']['mean'],1.)

if __name__=='__main__':
    unittest.main()
