# PREREG — Sharp addresses: does a nonlinear expansion fix Sol's 82%?

*Claude (Opus 5.5), 2026-10-19. Committed before the code exists and before any run.*

## Background

Sol's [AIbrainV2 Gate 2](https://github.com/anttiluode/AIbrainV2/blob/codex/tem-relational-addressing/docs/GATE2_RESULT.md)
(commit `0c234a3`) gives a 6×6 torus world a path-integrated phase key: 6 Fourier modes, 12 channels.
New-route recall with a delta memory is 82.2% at 18 visited locations and 56.7% at 30; untouched
retention after rewrites is 66.1% / 32.2%. An indexed coordinate table scores 100% everywhere.

My claim, made in conversation before any test: the gap is **key bandwidth**, not the memory rule.
Two phase keys have inner product k(Δ) = (1/6) Σ_ω cos(2π ω·Δ/6), a low-pass kernel, so neighbouring
locations share weight. The brain and TEM do not key memory on the grid code directly; they key it on
a sparse nonlinear expansion (place cells). Products of modes create the missing frequencies, so the
expansion should make keys near-orthogonal and close the gap to the table.

## Correction stated before the test

I also told Antti that noisy path integration is where a phase code would beat a table. That was
wrong as stated: a table that integrates the same noisy self-motion and corrects with the same sensory
evidence (a discrete Bayes filter) is the optimal estimator on a small known world, so the
representation alone gives no advantage there. The claim is withdrawn and not tested.

## What is run

Sol's code is vendored unchanged as `vendor/tem_addressing.py` (commit `0c234a3`), and his
`make_world`, routes, write schedule (3 repeats, β = 0.75), rewrite schedule and seeds 1000–1019 are
reused exactly. Only the key changes:

```
g   = Sol's 12-channel unit phase key
p   = topk_k( ReLU( W g ) ),  W ∈ R^{N×12} fixed Gaussian (seed 7), k winners kept, then unit-normalised
memory: Sol's delta writer on p
```

**Primary configuration (gating): N = 512, k = 16.**

## Gates (20 worlds, means)

| | criterion |
|---|---|
| S1 new-route recall, 18 visited | ≥ 95% |
| S2 new-route recall, 30 visited | ≥ 90% |
| S3 untouched retention after rewrite, 18 / 30 visited | ≥ 90% / ≥ 85% |
| S4 rewritten-value recall, 18 / 30 visited | ≥ 95% / ≥ 95% |
| S5 the **linear** control (same W, no ReLU, no top-k) stays within 5 points of Sol's phase + delta at 18 visited | the gain must come from the nonlinearity, not from width |
| S6 content key through the same expansion, 18 visited | ≤ 45% (aliasing cannot be expanded away) |

**PASS** = all six. Any miss is reported as such.

## Reported, not gated

Sweep N ∈ {64, 256, 1024} × k ∈ {4, 16, 64}; mean and max key cosine between distinct locations
before and after expansion; bytes of the memory **and** of W. The indexed table stays the reference.
Even a full pass does not beat it: it would show the structured memory *equals* a table once the address
is sharp, which is the point of the test, not a claim of superiority.
