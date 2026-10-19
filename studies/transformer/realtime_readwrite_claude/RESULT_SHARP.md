# Sharp addresses — PASS as pre-registered

*2026-10-19. Protocol: [PREREG_SHARP.md](PREREG_SHARP.md), committed before the code. Receipt:
[results/sharp_address.json](results/sharp_address.json). Reproduce in about 20 s on a CPU:
`python sharp_address.py`.*

Sol's Gate 2 world, routes, write and rewrite schedule, seeds and delta writer are used unchanged
(vendored at commit `0c234a3`; a test checks that this harness reproduces his numbers exactly). Only
the key changes: Sol's 12-channel phase key `g` goes through `p = unit(top16(ReLU(W g)))`, with `W` a
fixed 512×12 Gaussian.

## Result (20 worlds, means)

| visited | key | new route | rewritten | untouched |
|---|---|---:|---:|---:|
| 18 | Sol's phase + delta | 82.2% | 100% | 66.1% |
| 18 | **expanded phase + delta** | **100%** | **100%** | **99.6%** |
| 18 | same W, linear (no ReLU / top-k) | 82.2% | 100% | 66.8% |
| 18 | content key, same expansion | 34.2% | 72.5% | 19.6% |
| 18 | expanded phase + Hebb | 100% | 50.0% | 99.3% |
| 18 | indexed coordinate table | 100% | 100% | 100% |
| 30 | Sol's phase + delta | 56.7% | 98.6% | 32.2% |
| 30 | **expanded phase + delta** | **100%** | **100%** | **100%** |
| 30 | same W, linear | 56.8% | 98.6% | 31.5% |
| 30 | indexed coordinate table | 100% | 100% | 100% |

All six gates pass (S1–S6). The two controls carry the argument:

* **The linear control equals Sol's result to within 0.7 points.** A 512-wide linear map of a
  12-dimensional key still has rank 12 (tested), so width alone does nothing. The gain comes from the
  nonlinearity.
* **The content key stays aliased (34%).** The expansion sharpens an address; it cannot create one that
  the input does not carry.

Key geometry: the largest cosine between two distinct locations drops from 0.667 (phase) to 0.516
(expanded). That is not orthogonal, but the 36 expanded keys are linearly independent (rank 36, smallest singular value 0.47),
and the delta rule then stores each binding without disturbing the others.

The expanded Hebb row repeats Sol's stability–plasticity finding in a cleaner form. With sharp keys
both writers protect untouched memories (99%+). Only the delta rule can overwrite: Hebb on a
changed value reaches 50%, because the old association is still there at full strength.

Sweep (reported, not gated): even N = 64, k = 4 reaches 98.3% / 96.3% new-route recall. Dense
codes are worse than sparse ones at the same width (N = 256: k = 64 gives 76% untouched at 30 visited,
k = 4 gives 99%).

## What this shows and what it does not

* **It confirms the diagnosis.** Sol's 82% was a property of the key, not of the memory or of
  relational addressing itself: a 12-channel phase code is a low-pass address. A fixed random sparse
  expansion, the place-cell step in TEM and vector-HaSH, makes it effectively full-band.
* **It does not beat the table.** The structured memory now *equals* an indexed table, at a much higher
  cost: 16,384 bytes of memory plus 24,576 bytes of W, against the table's 54 bytes. On a small, known,
  noise-free world the table remains the right engineering answer.
* **It does not learn the structure.** The rotations are still given; that remains Sol's next gate.

## Why it matters for this repo

Gate 0 found a binding signal in Qwen's layer 14 that is real (41%, 3.3× chance) but not sharp
(negative margin). This result says what to do with a signal like that. Do not use it as a key
directly. Pass it through a sparse nonlinear expansion first, and let the delta memory write on the
expanded code. Gate 1's key will be designed that way, after the Gate 0b diagnostics are in.
