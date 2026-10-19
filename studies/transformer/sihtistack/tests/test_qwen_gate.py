"""Regression: token IDs (not city-name strings) index a 1-D logits tensor."""
import torch
import pytest
from sihtistack.qwen_gate import answer, margin


def test_city_margin_uses_token_ids():
    # Deliberately use sparse IDs unlike position in candidate dict.
    candidates = {"Oslo": 7, "Rome": 23, "Paris": 41}
    logits = torch.full((64,), -100.)
    logits[7] = 5.
    logits[23] = 3.
    logits[41] = -2.
    assert answer(logits, candidates) == "Oslo"
    assert margin(logits, candidates, "Oslo") == pytest.approx(2.)
    assert margin(logits, candidates, "Paris") == pytest.approx(-7.)


def test_city_margin_handles_wrong_city_and_unrelated_tokens():
    candidates = {"Oslo": 7, "Rome": 23, "Paris": 41}
    logits = torch.full((64,), -100.)
    logits[7], logits[23], logits[41] = 1., 6., 2.
    logits[0] = 100.  # Out-of-list token must not affect restricted candidate margin.
    assert answer(logits, candidates) == "Rome"
    assert margin(logits, candidates, "Oslo") == pytest.approx(-5.)
    assert margin(logits, candidates, "Rome") == pytest.approx(4.)
