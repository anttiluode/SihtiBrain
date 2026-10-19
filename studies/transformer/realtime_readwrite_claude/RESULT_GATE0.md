# Gate 0 result — FAIL as pre-registered

*Run 2026-10-19 by Antti on his machine (Windows, Python 3.13, float32). Receipt:
[results/gate0_qwen2.5-0.5b.json](results/gate0_qwen2.5-0.5b.json). Decision rule unchanged from
[PREREG.md](PREREG.md).*

| | value |
|---|---|
| layer chosen on dev worlds | **14** of 24 |
| test centred accuracy at layer 14 (1,200 queries) | **41.4%**, 95% Wilson [38.7%, 44.2%] |
| no-binding control at layer 14 | 12.9% (chance 12.5%) |
| 16-way: lands on target Ben / on own anchor Anna | 32.5% / 11.6% |
| mean margin cos(correct) − max cos(wrong) | −0.021 |
| **decision** (needs ≥ 80% and control ≤ 25%) | **FAIL** |

## What the numbers say

* **There is a real binding signal.** 41% is 3.3× chance with a tight interval, it is absent when the
  relation sentences are removed (12.9%), and the relational state is *not* simply about the anchor
  (only 11.6% land on Anna in the 16-way test). Something at "Anna's brother" points to Ben.
* **It is not a usable address.** The mean margin is negative: even when the right entity wins, it
  wins narrowly, and usually a wrong entity is closer. A memory keyed on this state would retrieve the
  wrong fact most of the time.
* **It lives in the middle of the network.** Accuracy is at chance through layer 9, rises from layer
  10, peaks at 14–15 (dev 0.417 / 0.405), and falls back towards 20% by the last layers, where the state
  is turning into a next-token prediction. Same shape as the "workspace" band in the J-space work, but
  that is an observation, not a test.

## What it does not settle

The pre-registered probe reads the state at the last token of `Anna's brother`, with no cue that the
referent is needed next, and compares whole-residual cosines. A failure there leaves three readings open:

1. the model does not resolve "Anna's brother" → Ben at all (a 0.5B model may not do the two-hop);
2. it resolves it, but at the *next* position, when it has to produce the referent;
3. it resolves it into a **subspace** (a binding ID in Feng & Steinhardt's sense) that full-residual cosine
   drowns in token identity ("brother" vs "Ben").

These are distinguished by the post-result diagnostics in `gate0b_diagnostics.py`, which are labelled
post-result and cannot change this decision.
