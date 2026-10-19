# Gate 1: two-cycle branch sampling

## Question and prediction

Can an identical scalar somatic ping steer across a two-branch cable using
only recent branch flux, and can successive soma echoes recover branch
distinctions that the passive soma cannot see? The useful claim is narrow:
local state must change the probed operator, and the probe schedule must
sample complementary orientations. A symmetric passive cable alone cannot
reveal hidden input through a same-operator pulse subtraction.

The attached [Vollan et al. (Nature 2025)](https://doi.org/10.1038/s41586-024-08527-1)
reports alternating spatial theta sweeps and tests a coverage-overlap model.
It does **not** establish the specific branch-fatigue rule below. The
[Gurnee et al. (2026) J-space preprint](https://arxiv.org/abs/2607.15495)
motivates an analogy about limited output access to distributed state, not
an identification between that transformer's Jacobian lens and our cable.

## Minimal mechanism

Seven nodes: soma `0`, roots `1,4`, leaves `2,3,5,6`. Each node leaks `0.15`,
both root wires begin at `0.8`, leaf wires are `0.5`.

```text
H(g) = 0.15 I + Σ_edges g_e b_e b_eᵀ
v = H(g)⁻¹ (leaf current + somatic pulse)
F_b = g_b (v_soma,ping − v_root_b,ping)
a_b(next) = 0.4 a_b + 0.6 F_b / (F_left + F_right)
g_b = 0.8 exp(clip(−24(a_b − mean(a)), −1.6, 1.6))
```

Each cycle is a *settled* cable response and then an adaptation update.
The pulse is the same on both sides and every cycle. The normalized update
means scaling its amplitude changes flux but not the chosen direction.
Initial `a=(0.500001, 0.499999)` supplies a known tiny seed; with perfectly
equal `a`, this deterministic system stays symmetric. The high gain and
clipping are explicit, phenomenological choices, not measured kinetics.

For the readout, we independently solve the **common passive** cable for
the current leaf pattern. Each root takes a local pre-ping voltage and
modifies its phase conductance:

```text
g_probe,b = g_phase,b [1 + 1.5 sigmoid(8(v_passive,root_b − 1.9))]
echo = (v_soma,with_ping − v_soma,without_ping) / pulse
```

Both terms of the echo use the **same latched operator**. This prevents a
changing unpinged baseline from masquerading as an echo. The state gate is
the nonlinear assumption that breaks the linear cable's blindness. The
alternating phase conductances are generated on a neutral zero-input sweep
and then frozen for the three-pattern readout. They cannot encode the
pattern's label. The reader knows which phase was presented.

## Observations and controls

All three noiseless patterns have the same passive soma voltage:

| Pattern | Four leaves | Echo in phase 1 | Echo in phase 2 |
|---|---|---:|---:|
| Left | `1,1,0,0` | 1.4450 | 1.5872 |
| Right | `0,0,1,1` | 1.5872 | 1.4450 |
| Split | `1,0,1,0` | 1.5686 | 1.5686 |

The closest two prototype pairs are about `0.125` apart in two-dimensional
echo space. At fixed `g`, a **linear** cable has exactly the same echo for
every input by superposition. A symmetric state-gated cable separates split
from clustered patterns but not left from right. A single asymmetric gate
can already distinguish all three noiseless patterns: the alternating
pair improves the noisy margin; it does not create a previously impossible
kind of observation. The externally preset paired control ties the adaptive
pair exactly. Thus this experiment does **not** establish adaptation as a
necessary mechanism.

With 4 noise seeds × 300 balanced cases, mean accuracies are 1.000
(adaptive), 1.000 (preset), 0.922 (two static), 0.678 (symmetric), and
0.321 (linear). Input noise is zero-sum, so passive soma remains matched.
The first 20 cycles are treated as a transient; signed branch contrast
flips each subsequent cycle. The raw receipt is [gate1_metrics.json](gate1_metrics.json).

## Limits and next falsification

- The branch gate is hand-defined from passive root voltage. This is a toy
  nonlinear observation law, not simulated ion channels or a real spike.
- The steep high-gain period-two cycle depends on the update, the gain and
  the known symmetry-breaking seed. It is not evidence that cortical
  alternation is produced this way.
- A reader must retain phase information; pair classification is a
  nearest-prototype lookup, not learned unsupervised object recognition.
- No slow write, segmentation, moving object, cortical column, or image
  model is demonstrated here. A 16-column moving-bar test needs causal
  neighbor delays, an object/noise baseline, a no-ping control, a static
  asymmetric schedule with equal pulses, held-out trajectories and no
  current-frame label sent in apical context. That is a separate gate.
