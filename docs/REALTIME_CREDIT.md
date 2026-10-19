# Gate 5: real-time credit, against a three-factor Hebbian rule

*Claude (Opus 5.5), 2026-10-19. The question this repository carries after the split: can a physical-style feedback event generate reliable credit despite distance, timing and noise? SihtiBrainV2 (operator memory) and SihtiBrainV3 (autonomous probing) are not touched.*

![Gate 5](realtime.png)

Reproduce: `python demo_realtime.py` (≈25 min CPU; `--replot` redraws) → `runs/realtime/`; tests: `python -m unittest tests.test_realtime_gate -v`. Numbers: `docs/realtime_metrics.json`.

## 1. What runs in real time now

Everything gates 3–4 scheduled by hand is replaced by dynamics, on the gate-4 tree with an 8-compartment trunk:

* **The soma fires its own pings**, an inhomogeneous Poisson process with rate `r(v₀) = σ(6(v₀ − 0.45))` per time unit; a ping is a somatic current pulse (amplitude 2, width 0.5).
* **Bursts**: each ping becomes a burst (two extra pulses) with probability `σ(4(v_tuft − θ_b))`, with θ_b set so the baseline is 0.3 at the free tuft voltage. This is the BAC-style coupling: the tuft decides how the soma's output looks.
* **Context arrives late, at the tuft**: a graded current `3·(y − v₀_free)` switched on after the early window, plus a 4-unit delay before the late window. It reaches the soma passively only as α ≈ 0.012 (gate 4), and much more through burst probability.
* **Synapses integrate calcium continuously**, `x_i · g(v_i(t))`, over an early window (before context) and a late window (after it), each 20 time units, after a 30-unit warm-up with pings running.
* **Evaluation**: training on amplitude-jittered patterns, testing on fresh held-out jitter draws (Sol's V2 standard).

Rules compared — they share the somatic-bias update (change in burst count), so only the synaptic credit differs:

| Rule | Synaptic update |
|---|---|
| ping-local | `[x_i>0] · log(Ca_late,i / Ca_early,i) / k` — local calcium, local state |
| three-factor | `x_i · (bursts_late − bursts_early) / 5` — presynaptic activity × one global signed signal |
| exact | implicit-function gradient of the somatic loss (upper bound) |

## 2. The sign theorem: why one cell does not need per-synapse credit signs

In the convex regime the cell's Jacobian H is symmetric, positive definite and has non-positive off-diagonals: a nonsingular **M-matrix**. Its inverse is entrywise positive. So every transfer impedance from the soma to a synapse is positive (minimum 0.60 over 324 checked states; a test), and

```
−∂C/∂w_i = −(v₀ − y) · x_i · [H⁻¹]_{0i}      with  [H⁻¹]_{0i} > 0.
```

**Every active synapse's gradient has the sign of the error.** Per-synapse credit in a single cell can only reweight magnitudes, never decide signs. A rule that gets one global signed bit per pattern and multiplies by presynaptic activity is already sign-correct. This is why the ternary code worked in gate 4, and it predicts that a three-factor rule should match the ping-local rule here.

## 3. Results

### A–B. Bias and variance of the updates (cluster task, probe states, 24 trials per pattern)

| Rule | Trial-averaged alignment | Single-trial alignment |
|---|---|---|
| ping-local | 0.82 – 0.97 | 0.66 – 0.82 |
| three-factor | 0.84 – 0.97 | 0.79 – 0.92 |
| ping-local, bursts off | −0.84 – +0.21 (noise around zero; see §4) | −0.16 – +0.05 |

Both rules are equally unbiased; the three-factor rule is *less* noisy per trial, because one global burst-count difference is a cleaner estimate than four separate calcium log-ratios.

### C. The cluster task cannot rank credit rules

Held-out accuracy (±5 % jitter, 30 patterns, 4 seeds):

| Condition | Held-out accuracy | Mean |
|---|---|---|
| exact gradient | 1.0, 1.0, 1.0, 1.0 | 1.00 |
| ping-local | 0.97, 0.67, 0.73, 0.90 | 0.82 |
| three-factor | 0.80, 0.73, 0.83, 0.97 | 0.83 |
| ping-local, bursts off | 0.33 ×4 | 0.33 |
| **ping-local, labels shuffled every epoch** | 0.97, 0.83, 0.80, 0.77 | **0.84** |
| **three-factor, labels shuffled every epoch** | 0.83, 0.83, 0.90, 0.83 | **0.85** |
| bias only (weights frozen) | 0.63, 0.67, 0.53, 0.57 | 0.60 |
| untrained | 0.33, 0.60, 0.50, 0.50 | 0.48 |

**Uninformative training does as well as informative training.** The branch nonlinearity is built in and symmetric: with equal weights, clustered inputs already drive the soma harder than split ones. Labels reshuffled every epoch average to a constant target, which pulls the four weights toward equal values and sets the offset — and that alone sorts the patterns. Bias-only training (0.60) shows that equalising the weights is the part that matters. The task measures whether learning reaches the cell's built-in operating point, not whether credit is assigned.

**Consequence for earlier results.** Gate 3's shuffled control used *one fixed* permutation of the labels — a consistently wrong teacher, which fails for a different reason — so its separation from the real teacher does not show that the two-ping rule used the labels beyond calibration. The alignment measurements of gates 2–3 (which compare update directions with the true gradient) are unaffected. SihtiBrainV2's gate 4 uses the same clustered-vs-split task, so its shuffled-target control should be checked with labels reshuffled every epoch.

### D. A task the cell cannot solve by itself: left cluster high, right cluster low

The symmetric nonlinearity cannot produce a left/right difference, and reshuffled labels cannot either. Held-out accuracy (4 seeds):

| Condition | Held-out accuracy | Margins |
|---|---|---|
| exact gradient | 1.0 ×4 | 0.21 – 0.30 |
| **ping-local** | **1.0 ×4** | **0.20 – 0.38** |
| **three-factor** | **1.0 ×4** | **0.21 – 0.32** |
| bias only | 1.0, 0.4, 0.5, 0.5 (one seed starts asymmetric) | |
| ping-local, reshuffled labels | 0.5 ×4 (chance) | |
| three-factor, reshuffled labels | 0.5, 0.5, 0.5, 1.0 (one lucky seed, margin 0.11) | |

**Real-time credit works:** with self-generated pings, a late tuft context, an 8-compartment trunk and decaying calcium integrals, the cell learns a task its architecture does not solve, and reshuffled labels fall to chance. **And the three-factor rule matches the ping-local rule exactly**, as the sign theorem predicts.

## 4. Three things found on the way (each changed the experiment)

* **Ping-driven drift.** Without a warm-up the early window starts at the ping-free equilibrium while the pings keep charging the slow soma (τ ≈ 7). Every late window then had more calcium than its early window, whatever the context said, and the ping-local rule *anti-learned* (held-out 0.13–0.33). A 30-unit warm-up removes it (test: with no context, the mean log ratio is zero within its error).
* **The jitter limit is the cell, not the rule.** With the exact gradient, held-out accuracy is 0.97–1.0 at ±5 % amplitude jitter, 0.90–0.93 at ±10 %, and 0.77–0.83 at ±20 %, at any trunk length. The clustered-vs-split signal from this mild branch nonlinearity is smaller than ±10 % input variation. Held-out evaluation therefore uses ±5 %.
* **Bursts off = below the noise floor, not wrong-signed.** With pings but no burst coupling, the passive context signal (mean update ≈ 0.0005; right sign, as a no-ping run shows deterministically) sits far below the Poisson ping noise (standard error ≈ 0.02–0.03 per pattern over 40 trials). The negative alignments for this condition are sampling noise.

## 5. Verdict

* Real-time credit **survives distance, delay and its own spiking**, but only through the burst code. Passive graded context over an 8-compartment trunk is lost in the cell's own ping noise (gate 4's result, now with self-generated noise instead of an added read noise).
* Inside one cell, **the ping-local rule does not beat a three-factor rule** — not on bias, not on variance, not on a task the cell must actually learn (both 4/4, same margins). That is not a weakness of the implementation; the M-matrix theorem says per-synapse credit cannot change signs in a single convex cell. The calcium machinery of gates 3–5 is a correct but unnecessary refinement *for one neuron*.
* **The clustered-vs-split task is the wrong benchmark for credit rules**: uninformative training reaches the same held-out accuracy. Any future comparison here, in V2 or V3, needs a task the architecture cannot solve by itself and labels reshuffled every epoch as the null.
* **Where per-synapse credit could matter: networks with signed connections.** Across cells, an inhibitory link flips the sign of a synapse's effect on the output, the M-matrix property no longer holds, and one global bit is no longer sign-correct for every synapse. That is the only place left where "ping with state" could beat a three-factor rule, and it is the stopping test for this repository: two or more cells with an inhibitory connection, held-out data, the three-factor rule as baseline. If the ping-local rule does not win there either, the single-cell credit story here is complete and the useful residue is the event-code / noise result (gates 4–5), which belongs with the physical-learning hardware line.

## 6. Ledger

* Spikes are Poisson pulses, not Hodgkin–Huxley; bursts are added pulses, not a simulated Ca²⁺ spike; burst probability is a sigmoid of tuft voltage.
* The context current is computed from the error (who computes it is not modelled).
* One task, one cell, four seeds, dimensionless units; Adam on the local updates.
* Burst threshold is re-anchored to the current free tuft voltage each trial (a moving baseline, as in burst-dependent plasticity); how a real cell sets that baseline is not modelled.
