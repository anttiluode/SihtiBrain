# Gate 4 — delayed apical context, shaped return ping, basal credit

This gate takes [SihtiBrainV2's first two-window experiment](../README.md) one step further: the earlier experiment **writes a cable operator** without targets; this one tests whether an *externally informed later cortical window* can train a basal readout through a state-shaped returning ping.

The six-panel figure is produced by the demo and uploaded as a GitHub Actions artifact (generated files are intentionally git-ignored).

The figure is generated locally/under GitHub Actions via `python demo_apical_credit.py`. Generated runs are intentionally git-ignored; download the CI artifact or run the script.

## 1. Information pathway and conditions

Nine compartments: the original seven-node symmetric basal tree, connected from soma 0 through trunk 7 to apical tuft 8. The stationary operator is reciprocal and passive:

```
H9 = g_leak*I + Σ_edge g_e b_e b_e^T
v_early = H9^(-1) I_basal         # target absent at this stage
transfer = e_soma^T H9^(-1) e_tuft = 0.2131569754
```

The fixed early-window basal feature is `φ_r(x)=sigmoid(10*(v_root_r−2))`, separately for left and right branch. This **chosen nonlinearity** is what makes the two same-branch patterns distinguishable from four cross-branch patterns; it is *not learned*. Trainable parameters are two basal-branch **readout synapses** `w=(w_left,w_right)` and a somatic bias `b`. The model predicts `ŷ=sigmoid(w·φ(x)+b)` with *no later information available at inference*.

Only after the prediction, context supplies the correct target `y` to the **apical tuft** as signed current `I_tuft=y−ŷ` (this is supervised error at the apical input, even though the basal synapses do not receive it). The first and later pulse amplitudes are:

```
p_early = 1
Δv_soma = transfer * I_tuft
p_later = 1 + 0.85*tanh(4*3*Δv_soma)  # hypothetical somatic gain
```

The long passive transfer is low; the gain 3 is a deliberately selected phenomenological apical-to-axon amplification, **not a BAC or NMDA simulator**.

Each basal-branch readout synapse sees only its earlier local eligibility `φ_r`, pulse strength attenuated at its location, and two positive calcium measurements:

```
Ca_r(p) = exp(0.3*v_root_r) * expm1(1.6*attenuation_r*p)
Δw_r = 0.11 * φ_r * log(Ca_r(p_later)/Ca_r(p_early))
Δb   = 0.11 * mean_r(log(Ca_later/Ca_early))
```

No input to the **local weight update** is an explicit target or scalar error. **But** the ratio changes because the pulse was shaped by an apical current encoding `y−ŷ`; this is information-carrying feedback, not learning from nowhere.

The two pulses and context windows are scheduled in this abstract gate. Delay is represented by a hard eligibility cutoff at 12 model steps, rather than an integrated kinetic trace. Autonomous spiking, voltage plateau dynamics, learned nonlinear features, and a genuine apical-to-basal bAP wave are **not** demonstrated.

## 2. Reproduction

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python demo_apical_credit.py --output runs/apical_credit --epochs 35
```

Four seeds; six input pairs; 12 training examples per pair (=72), 32 independent test examples per pair (=192), with separately seeded ±8% amplitude jitter. Same task/parameter choices across controls; 35 epochs of online training; common learning rate 0.11; first-window feature extractor frozen.

### Held-out accuracy after 35 epochs

| Condition | Seed 0 | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|---:|
| **Tuft → state-shaped ping → basal update** | **100%** | **100%** | **100%** | **100%** |
| No ping | 66.7% | 66.7% | 66.7% | 66.7% |
| No apical context | 66.7% | 66.7% | 66.7% | 66.7% |
| Shuffle targets between examples | 66.7% | 66.7% | 66.7% | 66.7% |
| Zero basal eligibility | 66.7% | 66.7% | 66.7% | 66.7% |
| Eligibility expired before context | 66.7% | 66.7% | 66.7% | 66.7% |
| Physically disconnect trunk | 66.7% | 66.7% | 66.7% | 66.7% |
| Replace sigmoid dendritic features with linear root voltages | 66.7% | 83.3% | 83.3% | 51.0% |
| **Ordinary direct-gradient learner** | **100%** | **100%** | **100%** | **100%** |

The majority-class floor is 66.7%, because four of the six pairs are split. The fixed-feature geometry makes this small task readily solvable; the direct-gradient baseline confirms it is **not** an improvement over conventional learning.

For the two-ping condition, all four held-out margins `min(score_cluster)−max(score_split)` are positive, spanning roughly 0.86–0.93. Accuracy alone could hide close calls, so the JSON records margins, binary cross-entropy and the learned weights.

### Noise matters when context becomes weak

With strong selected apical gain 3, 0%, 3%, and 10% multiplicative calcium read noise all retain 100% accuracy on these repeated toy examples. When the contextual gain is reduced to 0.05, mean held-out accuracy falls from ~97.7% noiseless to ~94.0% at 3% noise and ~91.4% at 10% noise. This does **not** contradict the severe weak-teacher noise problem measured by [SihtiBrain Gate 3](https://github.com/anttiluode/SihtiBrain/blob/claude/ping-as-nudge/docs/PING_WITH_STATE.md): the teaching strength, dynamical model, timescale, noise site and repeated sample count differ.

### Controls and causal limits

- A disconnected trunk eliminates the teaching route, even though the same basal task is present.
- Context shuffled independently of the current example does not train the correct mapping.
- Local eligibility is required for the two basal readout weights; without it, the somatic bias can move but cannot solve this task.
- A fixed outgoing pulse with identical early and late strength gives exactly zero calcium log-ratio even if the earlier basal voltages differ.
- Fractional calcium cancels the multiplicative baseline voltage factor in this chosen exponential readout.
- Expired eligibility blocks the write by **assumption**; a real biological decay time has not been measured here.
- The run compares a locally measured two-ping method to direct SGD, and therefore isolates an *information-routing possibility*, not a better learning algorithm.

## 3. Connection to the apical cortical diagram

In the time-window diagram, the apical tuft is a contextual query surface. **It is not an error-free source of labels.** In this gate, a later external target appears there; the soma's response to tuft input changes the next pulse; basal synapses combine that returning global event with local eligibility. That is a physically plausible *shape* of a feedback route, not evidence that cortex implements these exact equations.

Gates 1–2: local voltages can keep a distinction that the soma misses, and state-dependent cable modification can make it visible to a probe. Gate 4: when later evidence enters at a separate compartment, the returning pulse can mediate a useful learning update without sending explicit target bits to each basal site.

Previous relevant work: Guerguiev, Lillicrap & Richards (2017) apical teaching compartments; Larkum et al. (1999) apical-basal coincidence; Brunner & Szabadics (2016) analog modulation of backpropagating spikes; EqProp and three-factor plasticity models. None of their biological findings directly validates this gate's synthetic gain or instantaneous calcium model.

## 4. Next falsifiable gate

Put *continuous* soma threshold crossing, tuft-to-soma propagation, pulse travel time, synaptic calcium integration and trace decay into one time-stepped ODE. Require the entire 2-ping update to arise from those dynamics, rather than a static `p_later` function and a hard expiry threshold. Test held-out temporal sequences with no teacher leakage at inference, equal-compute nonlinear baselines, and explicit signal-to-noise budgets.

That is where the "ping keeps the ball rolling" intuition becomes a measurable causal dynamics claim rather than an engineered learning channel.
