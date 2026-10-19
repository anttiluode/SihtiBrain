"""CPU tests for Gate 1b / Gate 2."""
import os
import sys
import unittest
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gate1_twohop as g1  # noqa: E402
import gate2_speak as g  # noqa: E402


class TestWorld(unittest.TestCase):
    def test_every_anchor_and_relation_twice(self):
        for seed in range(20):
            w = g.make_world(seed, g1.DEV_NAMES)
            self.assertEqual(set(Counter(w.bind_anchor).values()), {2})
            self.assertEqual(set(Counter(w.bind_rel).values()), {2})
            self.assertEqual(len(set(zip(w.bind_anchor, w.bind_rel))), g.N)

    def test_spans(self):
        w = g.make_world(4, g1.DEV_NAMES)
        for i in range(g.N):
            a = w.anchors[w.bind_anchor[i]]
            self.assertEqual(w.chunk[w.anchor_end[i] - len(a):w.anchor_end[i]], a)
            self.assertEqual(w.chunk[w.rel_end[i] - len(w.ref(i)):w.rel_end[i]], w.ref(i))
            t = w.targets[i]
            self.assertEqual(w.chunk[w.bind_name_end[i] - len(t):w.bind_name_end[i]], t)
            self.assertEqual(w.chunk[w.fact_name_end[i] - len(t):w.fact_name_end[i]], t)
            c = w.cities[i]
            self.assertEqual(w.chunk[w.fact_city_end[i] - len(c):w.fact_city_end[i]], c)
            q, ae, re_ = g.question(w, i)
            self.assertEqual(q[ae - len(a):ae], a)
            self.assertEqual(q[re_ - len(w.ref(i)):re_], w.ref(i))
            for text in (q, g.answer_prompt(w, i)):
                self.assertNotIn(t, text)
                self.assertNotIn(c, text)


def planted_worlds(n_worlds, seed=0, d=48, noise=0.01):
    """Ideal states: the anchor state codes only the anchor, the phrase state only the relation."""
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_worlds):
        w = g.make_world(s, g1.DEV_NAMES)
        ea = {a: rng.normal(size=d) for a in w.anchors}
        er = {r: rng.normal(size=d) for r in set(w.bind_rel)}
        en = {t: rng.normal(size=d) for t in w.targets}
        ec = {c: rng.normal(size=d) for c in w.cities}
        L = len(g.LAYERS)
        st = lambda vs: np.repeat(np.array(vs)[:, None, :], L, axis=1) + noise * rng.normal(size=(g.N, L, d))
        A = [ea[w.anchors[w.bind_anchor[i]]] for i in range(g.N)]
        R = [er[w.bind_rel[i]] for i in range(g.N)]
        S = {"anchor_write": st(A), "anchor_query": st(A), "rel_write": st(R), "rel_query": st(R),
             "name_bind": st([en[t] for t in w.targets]), "name_fact": st([en[t] for t in w.targets]),
             "city_fact": st([ec[c] for c in w.cities])}
        out.append({"world": w, "S": S})
    return out


class TestKeys(unittest.TestCase):
    def test_only_the_conjunction_resolves_hop1(self):
        ws = planted_worlds(10)
        fit, test = ws[:7], ws[7:]
        hop = lambda cp, ex: g.acc([g.eval_1b(c, g.KeyBuilder(fit, cp, 14, 32, ex)) for c in test], "hop1")
        self.assertEqual(hop("outer", True), 1.0)
        self.assertGreaterEqual(hop("concat", True), 0.9)  # a ReLU of a sum is not a true conjunction
        # a linear concatenation of 4 anchor codes and 4 relation codes spans only 7 of the 8 pairs
        self.assertLessEqual(hop("concat", False), 0.75)
        self.assertLessEqual(hop("phrase", False), 0.5)
        self.assertLessEqual(hop("anchor", False), 0.5)
        two = g.acc([g.eval_1b(c, g.KeyBuilder(fit, "outer", 14, 32, True)) for c in test], "two_hop")
        self.assertEqual(two, 1.0)
        shuf = g.acc([g.eval_1b(c, g.KeyBuilder(fit, "outer", 14, 32, True), True) for c in test], "two_hop")
        self.assertEqual(shuf, 0.0)

    def test_memory_write_vectors_point_at_the_right_city(self):
        ws = planted_worlds(10)
        kb = g.KeyBuilder(ws[:7], "outer", 14, 32, True)
        mu = {L: np.mean(np.concatenate([c["S"]["city_fact"][:, g.LAYERS.index(L)] for c in ws[:7]]), 0)
              for L in g.LAYERS}
        for c in ws[7:]:
            m = g.write_vectors(c, kb, mu, 10, "memory")
            o = g.write_vectors(c, kb, mu, 10, "oracle")
            self.assertTrue(np.all(np.argmax(m @ o.T, axis=1) == np.arange(g.N)))
            wr = g.write_vectors(c, kb, mu, 10, "wrong")
            self.assertTrue(np.all(np.argmax(wr @ o.T, axis=1) != np.arange(g.N)))


class TestModelPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError as e:  # pragma: no cover
            raise unittest.SkipTest(str(e))
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=600, hidden_size=32, intermediate_size=64, num_hidden_layers=18,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)
        cls.model = Qwen2ForCausalLM(cfg).eval()
        cls.enc = staticmethod(lambda s: ([3 + (ord(c) % 590) for c in s], [(i, i + 1) for i in range(len(s))]))

    def test_injection_hook(self):
        texts = ["short one", "a somewhat longer prompt"]
        base = g.inject_logits(self.model, self.enc, 0, "cpu", texts, None, 10, 1.0, batch=2)
        zero = g.inject_logits(self.model, self.enc, 0, "cpu", texts, np.ones((2, 32)) / np.sqrt(32), 10, 0.0, batch=2)
        np.testing.assert_allclose(base, zero, atol=1e-6)
        pushed = g.inject_logits(self.model, self.enc, 0, "cpu", texts, np.ones((2, 32)) / np.sqrt(32), 10, 2.0, batch=2)
        self.assertGreater(float(np.abs(pushed - base).max()), 1e-3)
        single = g.inject_logits(self.model, self.enc, 0, "cpu", texts[:1], np.ones((1, 32)) / np.sqrt(32), 10, 2.0, batch=1)
        np.testing.assert_allclose(single[0], pushed[0], atol=1e-4)

    def test_end_to_end_tiny(self):
        first = lambda word: 3 + (ord(word[0]) % 590)
        res = g.run(self.model, self.enc, first, 0, "cpu", [0, 1], [2], [2000],
                    grid1b=[("outer", 14, 16, False), ("concat", 17, 16, True)], grid2=[(10, 1.0)],
                    batch=4, log=lambda *_: None)
        for gate in ("gate1b", "gate2"):
            self.assertIn(res[gate]["decision"], ("PASS", "FAIL"))
        for k in ("two_hop", "hop1", "no_memory", "shuffled_binding", "phrase_only_two_hop", "anchor_only_two_hop"):
            self.assertTrue(0 <= res["gate1b"][k] <= 1, k)
        for k in ("memory_write_way8", "oracle_write_way8", "wrong_write_right_city_way8"):
            self.assertTrue(0 <= res["gate2"][k] <= 1, k)


if __name__ == "__main__":
    unittest.main()
