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

RESULTS_PLACEHOLDER

## 4. Three things found on the way (each changed the experiment)

* **Ping-driven drift.** Without a warm-up the early window starts at the ping-free equilibrium while the pings keep charging the slow soma (τ ≈ 7). Every late window then had more calcium than its early window, whatever the context said, and the ping-local rule *anti-learned* (held-out 0.13–0.33). A 30-unit warm-up removes it (test: with no context, the mean log ratio is zero within its error).
* **The jitter limit is the cell, not the rule.** With the exact gradient, held-out accuracy is 0.97–1.0 at ±5 % amplitude jitter, 0.90–0.93 at ±10 %, and 0.77–0.83 at ±20 %, at any trunk length. The clustered-vs-split signal from this mild branch nonlinearity is smaller than ±10 % input variation. Held-out evaluation therefore uses ±5 %.
* **Bursts off = below the noise floor, not wrong-signed.** With pings but no burst coupling, the passive context signal (mean update ≈ 0.0005; right sign, as a no-ping run shows deterministically) sits far below the Poisson ping noise (standard error ≈ 0.02–0.03 per pattern over 40 trials). The negative alignments for this condition are sampling noise.

## 5. Verdict

* Real-time credit **survives distance, delay and its own spiking**, but only through the burst code. Passive graded context over an 8-compartment trunk is lost in the cell's own ping noise (gate 4's result, now with self-generated noise instead of an added read noise).
* Inside one cell, **the ping-local rule does not beat a three-factor rule**. That is not a weakness of the implementation; the M-matrix theorem says per-synapse credit cannot change signs in a single convex cell. The calcium machinery of gates 3–5 is a correct but unnecessary refinement *for one neuron*.
* **Where per-synapse credit could matter: networks with signed connections.** Across cells, an inhibitory link flips the sign of a synapse's effect on the output, the M-matrix property no longer holds, and one global bit is no longer sign-correct for every synapse. That is the only place left where "ping with state" could beat a three-factor rule, and it is the stopping test for this repository: two or more cells with an inhibitory connection, held-out data, the three-factor rule as baseline. If the ping-local rule does not win there either, the single-cell credit story here is complete and the useful residue is the event-code / noise result (gates 4–5), which belongs with the physical-learning hardware line.

## 6. Ledger

* Spikes are Poisson pulses, not Hodgkin–Huxley; bursts are added pulses, not a simulated Ca²⁺ spike; burst probability is a sigmoid of tuft voltage.
* The context current is computed from the error (who computes it is not modelled).
* One task, one cell, four seeds, dimensionless units; Adam on the local updates.
* Burst threshold is re-anchored to the current free tuft voltage each trial (a moving baseline, as in burst-dependent plasticity); how a real cell sets that baseline is not modelled.
