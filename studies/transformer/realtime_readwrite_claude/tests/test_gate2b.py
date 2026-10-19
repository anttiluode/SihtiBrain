"""CPU tests for Gate 2b."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import gate2_speak as g2  # noqa: E402
import gate2b_speak as g  # noqa: E402
from test_gate2 import planted_worlds  # noqa: E402


class TestGate2b(unittest.TestCase):
    def test_rehearsal_prompt_ends_before_city(self):
        w = g2.make_world(3, g.g1.DEV_NAMES)
        for i in range(g.N):
            r = g.rehearsal(w, i)
            self.assertTrue(r.endswith(f"{w.targets[i]} lives in"))
            self.assertEqual(r.count(w.cities[i]), 1)

    def test_memory_vectors_point_at_the_true_value(self):
        ws = planted_worlds(10)
        rng = np.random.default_rng(1)
        for c in ws:
            c["R"] = rng.normal(size=(g.N, len(g.VLAYERS), 48))
        kb = g2.KeyBuilder(ws[:7], "outer", 14, 32, True)
        mu = {L: np.mean(np.concatenate([c["R"][:, g.VLAYERS.index(L)] for c in ws[:7]]), 0) for L in g.VLAYERS}
        for c in ws[7:]:
            m, o = g.write_vectors(c, kb, mu, 17, "memory"), g.write_vectors(c, kb, mu, 17, "oracle")
            self.assertTrue(np.all(np.argmax(m @ o.T, axis=1) == np.arange(g.N)))
            wr = g.write_vectors(c, kb, mu, 17, "wrong")
            self.assertTrue(np.all(np.argmax(wr @ o.T, axis=1) != np.arange(g.N)))

    def test_end_to_end_tiny(self):
        try:
            import torch
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError as e:  # pragma: no cover
            raise unittest.SkipTest(str(e))
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=600, hidden_size=32, intermediate_size=64, num_hidden_layers=23,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)
        model = Qwen2ForCausalLM(cfg).eval()
        enc = lambda s: ([3 + (ord(ch) % 590) for ch in s], [(i, i + 1) for i in range(len(s))])
        first = lambda word: 3 + (ord(word[0]) % 590)
        r = g.run(model, enc, first, 0, "cpu", [0, 1], [2], [2000], grid=[(17, 1.0), (22, 2.0)],
                  hop1=("concat", 17, 16, True), batch=4, log=lambda *_: None)
        self.assertIn(r["decision"], ("PASS", "FAIL"))
        for k in ("memory_write_way8", "oracle_write_top1", "wrong_write_right_city_way8", "rehearsal_copy_top1",
                  "two_hop_same_keys"):
            self.assertTrue(0 <= r[k] <= 1, k)


if __name__ == "__main__":
    unittest.main()
