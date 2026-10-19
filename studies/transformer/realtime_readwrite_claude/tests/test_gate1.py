"""CPU tests for Gate 1. The model test uses a miniature random Qwen2 and a character tokenizer."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gate1_twohop as g  # noqa: E402


class TestWorld(unittest.TestCase):
    def test_spans_point_at_the_right_words(self):
        w = g.make_world(3, g.DEV_NAMES)
        for i in range(g.N):
            ref = f"{w.anchors[i]}'s {w.relations[i]}"
            self.assertEqual(w.chunk[w.rel_end[i] - len(ref):w.rel_end[i]], ref)
            b = w.targets[i]
            self.assertEqual(w.chunk[w.bind_name_end[i] - len(b):w.bind_name_end[i]], b)
            self.assertEqual(w.chunk[w.fact_name_end[i] - len(b):w.fact_name_end[i]], b)
            self.assertEqual(w.chunk[w.fact_dot[i] - 1], ".")
            self.assertIn(f"{ref} is {b}.", w.chunk)
            self.assertIn(f"{b} lives in {w.cities[i]}.", w.chunk)
            q, c = g.q_rel(w, i)
            self.assertEqual(q[c - len(ref):c], ref)

    def test_name_split_and_cities(self):
        self.assertFalse(set(g.DEV_NAMES) & set(g.TEST_NAMES))
        w = g.make_world(1000, g.TEST_NAMES)
        self.assertTrue(set(w.anchors + w.targets) <= set(g.TEST_NAMES))
        self.assertEqual(len(set(w.cities)), g.N)
        self.assertEqual(len(set(g.CITIES)), len(g.CITIES))

    def test_question_never_mentions_city_or_target(self):
        w = g.make_world(5, g.DEV_NAMES)
        for i in range(g.N):
            q, _ = g.q_rel(w, i)
            self.assertNotIn(w.cities[i], q)
            self.assertNotIn(w.targets[i], q)
            self.assertNotIn(w.cities[i], g.answer_prompt(w, i, False))


class TestMemory(unittest.TestCase):
    def test_delta_stores_orthogonal_pairs_exactly(self):
        M = g.Delta(16, 16)
        K = np.eye(16)[:8]
        V = np.eye(16)[8:]
        for k, v in zip(K, V):
            M.write(k, v)
        np.testing.assert_allclose(M.read(K[3]), V[3], atol=1e-12)

    def test_topk(self):
        x = np.array([0.1, 3.0, -2.0, 2.0, 0.5])
        y = g.topk_sparse(x, 2)
        self.assertEqual(int((y > 0).sum()), 2)
        self.assertAlmostEqual(float(np.linalg.norm(y)), 1.0)

    def test_planted_states_give_perfect_two_hop(self):
        """If the frozen states were ideal (same entity -> same vector), the pipeline must score 100%,
        and the shuffled-binding control must fall to 0."""
        rng = np.random.default_rng(0)
        d = 32
        worlds = []
        for s in range(6):
            w = g.make_world(s, g.DEV_NAMES)
            ent = {n: rng.normal(size=d) for n in w.anchors + w.targets}
            rel = [ent[a] + 0.5 * rng.normal(size=d) for a in w.anchors]
            nam = [ent[b] for b in w.targets]
            noise = lambda: 0.01 * rng.normal(size=(g.N, 1, d))
            S = {"rel_write": np.array(rel)[:, None] + noise(), "rel_query": np.array(rel)[:, None] + noise(),
                 "name_bind": np.array(nam)[:, None] + noise(), "name_fact": np.array(nam)[:, None] + noise(),
                 "name_query": np.array(nam)[:, None] + noise()}
            worlds.append({"world": w, "S": S})
        km = g.KeyMap(worlds[:4], 0, 16, False, g.MAIN_TYPES)
        for c in worlds[4:]:
            r = g.eval_world(c, km)
            self.assertEqual(r["two_hop"], g.N)
            self.assertEqual(r["one_hop"], g.N)
            self.assertEqual(g.eval_world(c, km, shuffle_binding=True)["two_hop"], 0)
            self.assertEqual(g.oracle(c), g.N)


class TestModelPath(unittest.TestCase):
    def test_end_to_end_tiny(self):
        try:
            import torch
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError as e:  # pragma: no cover
            raise unittest.SkipTest(str(e))
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=600, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)
        model = Qwen2ForCausalLM(cfg).eval()
        enc = lambda s: ([3 + (ord(c) % 590) for c in s], [(i, i + 1) for i in range(len(s))])
        first = lambda word: 3 + (ord(word[0]) % 590)
        res = g.run(model, enc, first, 0, "cpu", [0, 1], [2], [1000], grid=[(2, 8, False), (3, 8, True)],
                    layers=(2, 3), batch=4, log=lambda *_: None)
        self.assertIn(res["decision"], ("PASS", "FAIL"))
        self.assertEqual(res["test"]["n_questions"], g.N)
        self.assertEqual(res["test"]["oracle_table"], 1.0)
        for k in ("two_hop", "no_memory", "shuffled_binding", "in_context_ceiling", "one_hop",
                  "raw_keys_two_hop", "sentence_end_two_hop"):
            self.assertTrue(0.0 <= res["test"][k] <= 1.0, k)


if __name__ == "__main__":
    unittest.main()
