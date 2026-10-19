# Gate 2: the moving bar and phase cancellation

## Gate

Gemini's V3 suggestion asked whether two alternating branch pings could
"look ahead" along a moving object's path. We instantiate 16 seven-node
columns, each viewing a three-location patch of a noisy one-dimensional
frame. A bar of width 3 moves one column per frame and bounces at the edge.
The bar mask is withheld from the model and used only for evaluation.

A column's current local evidence comes from the passive somatic voltage of
leaves `[left pixel, center pixel, center pixel, right pixel]`. Its output
is sent to neighboring columns along each branch in two somatic pulse
phases. The **next** frame receives this prior before its sensory input.
The posterior mixes 70% current local evidence with 30% delayed context;
the next prior mixes 35% persistence with 65% neighbor routing. These
coefficients are fixed for every condition. No target or current-frame
ground truth travels on the apical route.

The branch fluxes come from the neutral adaptation sweep of Gate 1 and are
frozen for the movie. They are not updated by the bar, so this test does
**not** implement a slow image-written conductance basin or a physiological
apical circuit. There is no trained decoder.

## Exact cancellation

For a column output `q_i`, one phase routes a fraction `f_L` toward
`i-1` and `f_R` toward `i+1`. The next phase uses the mirrored fractions.
For this symmetric cable, `f_L¹=f_R²` and `f_R¹=f_L²`, with each phase
normalized to `f_L+f_R=1`. When the receiving column **averages the two**:

```text
context_j = (route(q, phase 1)_j + route(q, phase 2)_j) / 2
          = (q_(j−1) + q_(j+1)) / 2
```

This is ordinary symmetric neighbor smoothing. Alternation creates no
additional prediction information for a phase-blind receiver. The result
is independent of the movie, provided the two routed phases remain
mirrors with equal pulses. It does not prove that a phase-aware receiver
or a spatially varying operator would also cancel.

## Protocol and controls

- 32 fixed movie seeds: `41,97,127,211,303,509,701,907,1000..1023`.
  Each has 20 frames, 16 columns, width-3 bar, independent Gaussian
  sensor noise σ=0.35 and salt clutter probability 0.08.
- Adaptive: two phase fluxes from the neutral Gate 1 sweep, 36 cycles.
  Preset: identical phase conductances externally selected, same two pings.
  Static: first phase sent twice. Symmetric: equal left/right flux in both
  phases. All receive the same local evidence and the same update rule.
- Persistence and a simple velocity forecast see the same passive cable
  evidence. Velocity uses only the **two previous observed frames**, picks
  a displacement of −1, 0 or +1 by cross-correlation, then shifts the most
  recent evidence. It receives no target labels or future frame.
- Main measure: of the three highest-prior columns **before** frame `t`
  arrives, what fraction of the true bar's three columns are selected?
  The new-cell rank AUC measures whether the bar's newly entered column
  outranks previously empty negatives. Current-frame top-three recall is
  reported separately and benefits from seeing new sensory input.

| Condition | Next-bar top-three recall | New-cell rank AUC | Current-bar recall |
|---|---:|---:|---:|
| Adaptive two-phase | 0.615 | 0.860 | 0.908 |
| Preset two-phase | 0.615 | 0.860 | 0.908 |
| Symmetric two-pulse | 0.615 | 0.860 | 0.908 |
| Static asymmetric twice | 0.600 | 0.848 | 0.909 |
| Previous-frame persistence | 0.629 | 0.792 | 0.884 |
| Two-frame velocity | **0.781** | 0.883 | 0.884 |

Adaptive and symmetric next-frame priors differ by at most
`3.33×10⁻¹⁶` across the whole benchmark. The adaptive schedule therefore
**fails the tracking advantage gate**. The current-frame posterior is
slightly cleaner than raw evidence, but that improvement is equally
available to a symmetric, phase-blind neighbor smoother.
The velocity baseline wins next-bar recall on 31/32 movie seeds and ties
once; its mean paired advantage is 0.166 (bootstrap 95% interval
0.146–0.184 over these synthetic movie seeds; 10,000 resamples with seed
2026, recorded in the metrics JSON). This interval describes
this generator, not real image tracking.

Reproduce with `python demo_tracker.py --output runs/tracker`. The exact
per-seed receipt is [tracker_metrics.json](tracker_metrics.json), and the
[figure](tracker.svg) shows example seed 41. The repo test suite checks
the cancellation, delayed causality, motion geometry and control parity.

## Next gate, if pursued

Retain two separately tagged return channels at the receiver. Use only
past and current local evidence to estimate direction, then choose how to
gate those channels. Compare against the same explicit two-frame velocity
forecast, the phase-blind smoother, a preset two-phase schedule and a
static asymmetric two-pulse schedule. A phase-aware result must beat that
forecast on held-out motions to earn a tracking claim; a tie would only
show an alternative implementation of a known causal motion rule.
