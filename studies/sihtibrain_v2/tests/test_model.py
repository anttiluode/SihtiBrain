"""Deterministic controls for the SihtiBrainV2 read/write claim."""
import unittest
import numpy as np
from sihtibrain_v2 import (
    operator, currents, encode, passive_trajectory, shaped_ping,
    echo_with_operator, local_gate, probe, write, reset_and_read,
    context_from_neighbor, live_loop)

C = [1, 1, 0, 0]
S = [1, 0, 1, 0]


class PassiveTests(unittest.TestCase):
    def test_symmetric_positive_cable(self):
        h = operator()
        np.testing.assert_allclose(h, h.T, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(h).min(), 0.)

    def test_voltage_is_exact_equilibrium(self):
        en = encode(C)
        np.testing.assert_allclose(operator() @ en.voltage, en.leaf_drive, atol=1e-12)

    def test_soma_is_blind_to_local_arrangement(self):
        c, s = encode(C), encode(S)
        self.assertLess(abs(c.voltage[0] - s.voltage[0]), 1e-12)
        self.assertGreater(abs(c.residue[0] - s.residue[0]), .5)

    def test_constant_input_passive_energy_descends(self):
        for stim in (C, S):
            path, energy = passive_trajectory(stim)
            self.assertLessEqual(float(np.diff(energy).max()), 1e-10)
            self.assertGreater(np.linalg.norm(path[-1]), 0.)

    def test_cable_residues_telescope_exactly(self):
        path, _ = passive_trajectory(C, steps=120, dt=.025)
        h, drive = operator(), currents(C)
        residuals = np.array([.025*(h @ v - drive) for v in path[:-1]])
        np.testing.assert_allclose(path[0] - path[-1], residuals.sum(axis=0), atol=1e-12)

    def test_cable_residue_includes_leak_and_input(self):
        path, _ = passive_trajectory(C, steps=120, dt=.025)
        h, drive = operator(), currents(C)
        total = .025*(h@path[30] - drive)
        axial_only = .025*((h - .15*np.eye(7))@path[30])
        self.assertGreater(np.linalg.norm(total - axial_only), .01)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.c, self.s = encode(C), encode(S)
        self.p = shaped_ping(self.c.voltage[0])

    def test_identical_soma_history_gives_identical_sender_ping(self):
        self.assertAlmostEqual(self.p, shaped_ping(self.s.voltage[0]), 12)

    def test_equal_frozen_linear_operator_ping_blind(self):
        c = echo_with_operator((.8,.8), self.c.leaf_drive, self.p)[0]
        s = echo_with_operator((.8,.8), self.s.leaf_drive, self.p)[0]
        self.assertAlmostEqual(c, s, 12)

    def test_context_plus_branch_state_changes_echo(self):
        c = probe(self.c, apical=(1,0), ping=self.p)
        s = probe(self.s, apical=(1,0), ping=self.p)
        self.assertGreater(abs(c.echo - s.echo), .08)
        self.assertGreater(c.gate[0], s.gate[0])

    def test_no_apical_closes_fast_and_slow_gate(self):
        for en in (self.c, self.s):
            np.testing.assert_allclose(probe(en,apical=(0,0),ping=self.p).fast_g, [.8,.8],atol=1e-14)
            np.testing.assert_allclose(write(en,apical=(0,0),ping=self.p), [.8,.8],atol=1e-14)

    def test_no_ping_blocks_writer(self):
        for en in (self.c, self.s):
            np.testing.assert_allclose(write(en,ping=0.), [.8,.8],atol=1e-14)

    def test_erased_local_trace_destroys_differential_echo(self):
        c = probe(self.c,ping=self.p,trace_override=np.zeros(2)).echo
        s = probe(self.s,ping=self.p,trace_override=np.zeros(2)).echo
        self.assertAlmostEqual(c,s,12)

    def test_slow_write_is_visible_after_activity_reset(self):
        gc, gs = write(self.c,ping=self.p), write(self.s,ping=self.p)
        self.assertGreater(gc[0] - gs[0], .2)
        self.assertGreater(abs(reset_and_read(gc) - reset_and_read(gs)), .03)

    def test_apical_side_is_external_not_known_from_current_soma(self):
        left = encode([1,1,0,0])
        right = encode([0,0,1,1])
        self.assertLess(abs(left.voltage[0] - right.voltage[0]), 1e-12)
        self.assertGreater(probe(left,apical=(1,0),ping=self.p).gate[0], .9)
        self.assertGreater(probe(right,apical=(0,1),ping=self.p).gate[1], .9)


class ContextTests(unittest.TestCase):
    def test_previous_neighbor_provides_same_context_for_both_current_inputs(self):
        apical = context_from_neighbor([1,1,0,0])
        self.assertGreater(apical[0], .99)
        self.assertLess(apical[1], .01)
        self.assertGreater(abs(probe(encode(C),apical,ping=1).echo
                               -probe(encode(S),apical,ping=1).echo), .05)

    def test_shunt_blocks_apical_port(self):
        a = context_from_neighbor([1,1,0,0],inhibit=(1.,0.))
        self.assertAlmostEqual(a[0], 0.)
        self.assertAlmostEqual(local_gate(encode(C),a,1.)[0], 0.)


class LiveLoopTests(unittest.TestCase):
    def test_return_delay_cannot_change_prearrival_activity(self):
        en = encode(C)
        gated = live_loop(en,apical=(1,0),writer_eta=0.)
        passive = live_loop(en,apical=(0,0),writer_eta=0.)
        np.testing.assert_allclose(gated["voltage"][:44],passive["voltage"][:44],atol=1e-12)
        self.assertGreater(np.max(abs(gated["voltage"]-passive["voltage"])), .02)

    def test_frozen_writer_has_no_stored_difference(self):
        c = live_loop(encode(C),writer_eta=0.)
        s = live_loop(encode(S),writer_eta=0.)
        np.testing.assert_allclose(c["slow_g"][-1],s["slow_g"][-1],atol=1e-14)

    def test_live_writer_retains_a_difference_after_reset(self):
        c = live_loop(encode(C),writer_eta=.3)
        s = live_loop(encode(S),writer_eta=.3)
        self.assertGreater(c["slow_g"][-1,0] - s["slow_g"][-1,0], .03)
        self.assertGreater(abs(reset_and_read(c["slow_g"][-1])
                               -reset_and_read(s["slow_g"][-1])), .005)


if __name__ == "__main__":
    unittest.main()
