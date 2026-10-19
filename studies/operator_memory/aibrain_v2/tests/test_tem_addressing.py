"""Deterministic controls for fixed-structure relational memory."""
import unittest
import numpy as np
from tem_addressing import (
    SIDE,DIM,MEMORY_BYTES,PhaseIntegrator,CoordinateIntegrator,DeltaMemory,
    IndexedCoordinate,ObservedGraph,NoMemory,make_world,canonical_path,
    different_path,content_key,time_key,run_world,address_invariants,benchmark)

class RelationalAddressTests(unittest.TestCase):
    def test_all_grid_addresses_unique(self):
        r=address_invariants()
        self.assertTrue(r['passes'])
        self.assertGreater(r['minimum_distinct_address_distance'],.5)

    def test_closed_loop(self):
        a=PhaseIntegrator().follow('ENWS')
        b=PhaseIntegrator().key()
        np.testing.assert_allclose(a,b,atol=1e-12)

    def test_different_route_same_address(self):
        for x in range(SIDE):
            for y in range(SIDE):
                a=canonical_path((x,y))
                b=different_path((x,y))
                self.assertNotEqual(a,b)
                np.testing.assert_allclose(PhaseIntegrator().follow(a),
                                           PhaseIntegrator().follow(b),atol=1e-12)
                self.assertEqual(CoordinateIntegrator().follow(a),
                                 CoordinateIntegrator().follow(b))

    def test_phase_state_tracks_only_actions(self):
        p=PhaseIntegrator()
        p.follow(['E','N','E','S','W'])
        self.assertEqual(p.key().shape,(DIM,))
        self.assertAlmostEqual(np.linalg.norm(p.key()),1.)

    def test_equal_byte_fast_weights(self):
        self.assertEqual(content_key(3).shape,time_key(14).shape)
        for mode in ('phase','content','time','wrong_action'):
            self.assertEqual(DeltaMemory(mode).byte_count,MEMORY_BYTES)

    def test_wrong_action_breaks_path_invariance(self):
        m=DeltaMemory('wrong_action')
        a=m.key(canonical_path((2,3)),0)
        b=m.key(different_path((2,3)),0)
        self.assertGreater(np.linalg.norm(a-b),.2)

    def test_phase_key_recovers_different_path(self):
        m=DeltaMemory('phase')
        a=canonical_path((2,3));b=different_path((2,3))
        for _ in range(3):m.write(a,0,5)
        self.assertEqual(m.predict(b,3),5)

    def test_delta_rewrite_changes_a_binding(self):
        m=DeltaMemory('phase')
        a=canonical_path((1,4));b=different_path((1,4))
        for _ in range(3):m.write(a,1,3)
        self.assertEqual(m.predict(b,0),3)
        for _ in range(3):m.write(a,1,6)
        self.assertEqual(m.predict(b,0),6)

    def test_indexed_action_only_baseline(self):
        m=IndexedCoordinate()
        a=canonical_path((2,4));b=different_path((2,4))
        m.write(a,0,5);self.assertEqual(m.predict(b,1),5)
        m.write(b,2,7);self.assertEqual(m.predict(a,0),7)
        self.assertEqual(m.byte_count,3)

    def test_edge_only_graph_cannot_cross_unobserved_edge(self):
        m=ObservedGraph();a=canonical_path((2,2))
        m.write(a,0,5)
        self.assertEqual(m.predict(a,0),5)
        self.assertEqual(m.predict(different_path((2,2)),0),-1)

    def test_zero_memory_negative(self):
        m=NoMemory();m.write('EN',0,6)
        self.assertEqual(m.predict('NE',0),0)
        self.assertEqual(m.byte_count,0)

    def test_world_reproducible_not_same_as_next_seed(self):
        a,b,c=make_world(1000),make_world(1000),make_world(1001)
        for i in (1,2,3):np.testing.assert_array_equal(a[i],b[i])
        self.assertFalse(np.array_equal(a[2],c[2]))

    def test_run_reproducible_and_budgets(self):
        a=run_world(1000,18);b=run_world(1000,18)
        self.assertEqual(a,b)
        self.assertEqual(a['bytes']['phase_delta'],384)
        self.assertEqual(a['bytes']['indexed_coordinate'],54)
        self.assertEqual(a['never_visited']['indexed_coordinate'],0.)

    def test_strong_classical_baseline(self):
        r=benchmark(range(1000,1002))
        self.assertTrue(r['decisions']['gate0'])
        self.assertEqual(r['summary']['18']['indexed_coordinate']
                         ['first_new_route']['mean'],1.)
        self.assertFalse(r['decisions']['algorithmic_advantage_over_indexed'])

if __name__=='__main__':unittest.main()
