"""CPU tests for Gate 0. No download: the model test uses a miniature random Qwen2 and a stub tokenizer.

    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gate0_route_probe as g  # noqa: E402


class TestWorlds(unittest.TestCase):
    def test_deterministic_and_distinct(self):
        a, b = g.make_world(7), g.make_world(7)
        self.assertEqual(a.context, b.context)
        self.assertEqual(len(set(a.anchors + a.targets)), 16)
        self.assertNotEqual(g.make_world(7).context, g.make_world(8).context)

    def test_prompts(self):
        w = g.make_world(3)
        for k, v in w.prompts.items():
            self.assertEqual(len(v), g.N_BIND, k)
        for a, r, b in zip(w.anchors, w.relations, w.targets):
            self.assertIn(f"{a}'s {r} is {b}.", w.context)
        self.assertTrue(w.prompts["relational"][0].endswith(f"{w.anchors[0]}'s {w.relations[0]}"))
        self.assertTrue(w.prompts["direct_target"][0].endswith(w.targets[0]))

    def test_nobind_context_has_no_relations(self):
        w = g.make_world(5)
        for r in g.RELATIONS:
            self.assertNotIn(r, w.nobind_context)
        for name in w.anchors + w.targets:
            self.assertIn(name, w.nobind_context)

    def test_seed_sets_disjoint(self):
        self.assertFalse(set(g.DEV_SEEDS) & set(g.TEST_SEEDS))


class TestMetrics(unittest.TestCase):
    def test_perfect_and_shared_offset(self):
        rng = np.random.default_rng(0)
        D = rng.normal(size=(8, 64)) + 50.0  # big shared offset: raw cosine is uninformative
        R = D + 0.05 * rng.normal(size=D.shape)
        self.assertEqual(g.retrieval(D, R, centre=True)["n_correct"], 8)

    def test_random_is_near_chance(self):
        rng = np.random.default_rng(1)
        k = sum(g.retrieval(rng.normal(size=(8, 64)), rng.normal(size=(8, 64)))["n_correct"] for _ in range(400))
        self.assertLess(abs(k / 3200 - 1 / 8), 0.03)

    def test_way16_anchor_detection(self):
        rng = np.random.default_rng(2)
        T, A = rng.normal(size=(8, 32)), rng.normal(size=(8, 32))
        r = g.retrieval16(T, A, A + 0.01 * rng.normal(size=A.shape))
        self.assertEqual(r["n_own_anchor"], 8)
        self.assertEqual(r["n_target"], 0)

    def test_wilson(self):
        lo, hi = g.wilson(960, 1200)
        self.assertTrue(lo < 0.8 < hi)


class TestModelPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError as e:  # pragma: no cover
            raise unittest.SkipTest(f"torch/transformers missing: {e}")
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=600, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
        cls.model = Qwen2ForCausalLM(cfg).eval()
        cls.encode = staticmethod(lambda s: [3 + (ord(c) % 590) for c in s])  # stub character tokenizer
        cls.torch = torch

    def test_padding_invariance(self):
        texts = ["short", "a considerably longer prompt here", "mid length one"]
        batched = g.last_token_states(self.model, self.encode, texts, pad_id=0, device="cpu", batch=3)
        for i, t in enumerate(texts):
            single = g.last_token_states(self.model, self.encode, [t], pad_id=0, device="cpu", batch=1)
            np.testing.assert_allclose(batched[i], single[0], atol=1e-4)
        self.assertEqual(batched.shape, (3, 3, 32))  # embeddings + 2 layers

    def test_end_to_end(self):
        res = g.run(self.model, self.encode, 0, "cpu", [0, 1], [1000, 1001], batch=8, log=lambda *_: None)
        self.assertIn(res["decision"], ("PASS", "FAIL"))
        self.assertEqual(len(res["test_by_layer"]), 3)
        self.assertEqual(res["test_at_l_star"]["n"], 16)
        for a in res["test_by_layer"]:
            for k in ("acc_centred", "acc_raw", "acc_nobind", "way16_target", "way16_own_anchor"):
                self.assertTrue(0.0 <= a[k] <= 1.0)


class TestDiagnostics(unittest.TestCase):
    def test_ridge_recovers_a_planted_map(self):
        import gate0b_diagnostics as d
        rng = np.random.default_rng(3)
        M = rng.normal(size=(16, 16))
        worlds = []
        for _ in range(12):
            T = rng.normal(size=(8, 1, 16))
            worlds.append({"T": T, "R": T @ np.linalg.inv(M) + 0.01 * rng.normal(size=T.shape)})
        W = d.fit_ridge(worlds[:8], "R", "T", 0, 1e-4)
        self.assertEqual(d.acc_mapped(worlds[8:], "R", "T", 0, W), 1.0)

    def test_ridge_survives_identical_queries(self):
        import gate0b_diagnostics as d
        rng = np.random.default_rng(4)
        worlds = [{"T": rng.normal(size=(8, 1, 16)), "R": np.ones((8, 1, 16))} for _ in range(4)]
        W = d.fit_ridge(worlds, "R", "T", 0, 1.0)  # centred queries are all zero
        self.assertTrue(np.all(np.isfinite(W)))
        self.assertLessEqual(d.acc_mapped(worlds, "R", "T", 0, W), 1.0)

    def test_end_to_end_tiny(self):
        try:
            import torch
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError as e:  # pragma: no cover
            raise unittest.SkipTest(str(e))
        import gate0b_diagnostics as d
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=600, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
        model = Qwen2ForCausalLM(cfg).eval()
        enc = lambda s: [3 + (ord(c) % 590) for c in s]
        res = d.run(model, enc, lambda name: enc(name)[0], 0, "cpu", [0, 1], [2], [1000], batch=8,
                    log=lambda *_: None)
        self.assertTrue(res["post_result"])
        self.assertEqual(len(res["by_layer"]), 3)
        for k in ("top1_vocab", "way8", "nobind_way8"):
            self.assertTrue(0 <= res["A_behaviour"][k] <= 1)


if __name__ == "__main__":
    unittest.main()
