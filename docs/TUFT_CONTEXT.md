# Gate 4: context at the tuft

*Claude (Opus 5.5), 2026-10-19. Continues gate 3 on this branch. SihtiBrainV2 (Sol) is the unsupervised two-window line and is not modified; this gate is complementary to it.*

![Gate 4](tuft.png)

Reproduce: `python demo_tuft.py` (≈10 min CPU; `--replot` redraws) → `runs/tuft/`; tests: `python -m unittest tests.test_tuft_gate -v`. Numbers: `docs/tuft_metrics.json`.

Gate 3 put the context (the "later window", the teacher) at the soma. In a pyramidal cell the context arrives at the apical tuft, at the far end of a trunk. This gate adds that second zone to the same basal tree and asks what reaches the basal synapses.

```
tuft T ── trunk n_L ── … ── trunk n_1 ── soma 0 ── basal tree (gates 1–3: roots 1, 4; leaves 2, 3, 5, 6)
             ^ optional regenerative hotspot at the top of the trunk
```

Same units, leak (0.15), basal regenerative roots (gain 0.27), and two-ping fractional-calcium rule as gate 3. Trunk segments have axial conductance 0.5.

## 1. The soma is the only door

Order the compartments as basal B, soma 0, apical A. The soma separates them, so the Jacobian H (symmetric; the nonlinearities are local) has no B–A block. Block elimination gives, with s the Schur complement at the soma,

```
[H⁻¹]_{B0} = −H_BB⁻¹ h_B0 / s,   [H⁻¹]_{0A} = −h_0A H_AA⁻¹ / s,   [H⁻¹]_{00} = 1/s
[H⁻¹]_{BA} =  H_BB⁻¹ h_B0 h_0A H_AA⁻¹ / s  =  [H⁻¹]_{B0} [H⁻¹]_{0A} / [H⁻¹]_{00}
```

So context injected anywhere apical moves every basal synapse **exactly as if it had been injected at the soma**, scaled by one number,

```
α = [H⁻¹]_{0T} / [H⁻¹]_{00}.
```

Measured residual of the identity: ≤ 2.4 × 10⁻¹⁶, at nonlinear states. Through the full calcium readout, tuft-context updates equal α × soma-context updates synapse by synapse (test). This is the tree Green's-function factorisation through a cut vertex, not new mathematics. Its consequence here is sharp:

**Apical context cannot change which basal synapses get credit, or their relative shares. It sets one scalar.** The whole apical world reaches the basal tree as one number per moment. That is "the soma summates" read backwards.

## 2. The scalar collapses with distance

| Trunk compartments | 0 | 1 | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| α (passive) | 0.769 | 0.503 | 0.305 | 0.106 | 0.036 | 0.012 | 0.0014 | 0.0002 |
| α (best single hotspot, monostable) | — | 0.637 | 0.501 | 0.214 | 0.124 | 0.068 | 0.0091 | 0.0010 |

Without noise the direction stays exact at every length: alignment with the true gradient 0.989–0.992. With calcium read noise (alignment, mean of 4 probe seeds):

| Trunk | noise 0 | 1 % | 3 % |
|---|---|---|---|
| 0 | 0.989 | 0.960 | 0.799 |
| 2 | 0.992 | 0.890 | 0.560 |
| 4 | 0.992 | 0.568 | 0.231 |
| 6 | 0.992 | 0.235 | 0.088 |
| 8 | 0.992 | 0.088 | 0.037 |

The table depends only on noise / α: two more trunk compartments (α ÷ ≈3) at 1 % noise score the same as the shorter trunk at 3 % noise (0.235 vs 0.231; 0.088 vs 0.088).

## 3. One hotspot cannot fix it

A single regenerative compartment at the top of the trunk raises α by 3–6× on a grid of gain and threshold (the bump at 3 compartments is a grid point close to the fold). Its gain diverges only in a narrow sliver at the fold, and past the fold the trunk is bistable: an all-or-none event. To beat exponential decay over a long trunk the gain has to be distributed along it, and a distributed regenerative trunk is a propagating event. **The physics forces long-range apical context to travel as an event, and only the event's code survives.**

## 4. Which event code carries credit

The event is modelled as arriving at the soma at full size (a regenerative trunk is assumed, not simulated), with size 0.05 and a dead zone of 0.06 in error. Eight-compartment trunk:

| Context code | Alignment (noise 0 / 1 % / 3 %) | Training, 1 % noise: separate (margin > 0) / calibrated 6/6 | Margins |
|---|---|---|---|
| graded at soma (ideal reference) | 0.993 / 0.980 / 0.898 | 4/4 / 4/4 | +0.076 – +0.101 |
| graded at tuft, passive trunk | 0.992 / 0.088 / 0.037 | 0/4 / 0/4 | −0.476 – −0.016 |
| **ternary event: burst / none / pause** | **0.951 / 0.948 / 0.926** | **4/4 / 3/4** | **+0.082 – +0.091** |
| one-sided burst only | 0.573 / 0.563 / 0.511 | 0/4 / 0/4 | −1.447 – −0.503 |

* The ternary event loses about 5 % of alignment to quantisation and then **ignores read noise**: its size does not shrink with distance, so the signal stays far above the noise.
* It separates the classes as well as the ideal graded context (margins +0.082 to +0.091 vs +0.076 to +0.101). Calibration is weaker (3/4 at 6/6): a sign-only code minimises absolute rather than squared error, and when the trunk-loaded cell cannot reach both targets that optimum is flat, so outputs settle anywhere along it.
* **One-sided bursts fail.** With no way to say "less", the outputs can only be pushed up.

**Biological reading (hedged).** A ternary code relative to a baseline is what burst-dependent plasticity uses: bursts above a slow moving-average burst rate potentiate, single spikes below it depress (Payeur, Guerguiev, Zenke, Richards & Naud, Nat. Neurosci. 2021). The positive event matches BAC firing (Larkum, Zhu & Sakmann 1999). The negative channel needs something that suppresses tuft-driven bursting below baseline; tuft dendritic signals are under the control of deep-layer inhibitory circuitry (Murayama et al., Nature 2009), which is the natural candidate. This gate does not model any of these mechanisms; it shows which code a long trunk requires.

## 5. What this says about the next architecture

Section 1 is a constraint on a *single* cell: through its own cable, the apical tuft can tell the basal tree only one number at a time. Branch-specific context — telling one basal branch "yes" and another "no" in the same moment — cannot come through the soma. It needs separate input lines: oblique dendrites with their own context, other cells, other columns. That is an argument *for* the multi-column V3 design: per-branch context has to be distributed across cells.

## 6. Ledger

* The regenerative trunk is assumed, not simulated: the event reaches the soma at full size. A real Ca²⁺ spike's propagation, failure rate and timing are not modelled.
* The context current is computed from the error (context neurons are assumed to compare the cell's output with a target). Where that comparison happens is not modelled.
* Noise is only in the calcium read; voltages are noise-free.
* One task, 4 seeds, dimensionless units, trunk compartments of arbitrary size.
* The hotspot "best α" comes from a finite grid of gains and thresholds.

## 7. Next

The open step from gate 3 remains: real time. Self-generated pings at state-dependent rates, a delayed context event, a decaying calcium trace, and the ternary code expressed as burst rate relative to a moving baseline rather than as a hand-set sign.
