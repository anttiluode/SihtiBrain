# Inhibitory Sensitivity Gate — preregistered test (2026-10-21)

## Question
Can an **unlabelled, small inhibitory perturbation** distinguish two local states which have identical baseline output but occupy opposite sides of a nonmonotonic response curve? Is inhibition uniquely helpful, or does a matched excitatory probe expose the same information?

This is a deliberately abstract local response experiment, **not** a Gidon dCaAP biophysical model, a simulation of cortical circuits, or a demonstration of cheaper adaptive computation. The inspiration is the nonmonotonic dendritic transfer reported by Gidon et al. (Science, 2020), with separate hypothetical application to active observability.

## Frozen toy protocol, before confirmatory results

- Hidden input `I=1±u`; each paired left/right state uses the same `u`, drawn uniformly from `[0.35,0.85]`. Classes are balanced per batch and shuffled; the reader never receives `I`, `u`, or a class-dependent address.
- Smooth, symmetric illustrative response `A(I)=exp(-(I-1)^2)`. This *guarantees exactly matched noiseless outputs* for `1-u` and `1+u`. This analytically constructed equality is a limitation, not an empirical discovery.
- Small perturbation `eps=0.10`. Inhibitory: `I -> I-eps`, excitatory: `I -> I+eps`. No inhibition-specific mechanism is hardcoded. The reader gets only baseline and post-probe noisy scalar outputs.
- Independent Gaussian measurement noise at `sigma=0.015` (primary) and `0.04, 0.08` (stress); deterministic source seeds `(41,97,127,211)`, `800` trials/seed/noise. Baseline noisy scalar is shared across equally priced probe arms for paired fairness; every post-probe reading has independent noise.
- **Frozen, train-free decoder:** increased response after inhibition => overdriven, else weak; decreased response after excitation => overdriven, else weak. The sham two-read arm uses a no-op perturbation and the same inhibitory sign decoder. A passive-only arm uses baseline `y>0.75` as an intentionally simple diagnostic; all passive values and baseline noise have identical conditional distributions across labels. An input-current oracle has privileged information and is labelled as such, never ranked as a fair competitor.
- **Cost:** every two-read arm pays **two complete evaluations** of `A` plus two scalar measurements, compared with one for passive. Nothing is cached or counted as free. No adaptive-compute savings are asserted.

## Locked criteria, in this order
1. **Mechanism G1 (observable regime):** exact noiseless matched baseline (max error <=1e-12), and inhibitory probe gives strictly opposite signed differences for weak and overdriven states throughout tested `u`. At primary noise, mean inhibitory accuracy across seeds >=0.95, sham <=0.60, and inhibitory minus sham >=0.30.
2. **Distinctive inhibition G2:** inhibition must beat equally costly excitation by >=0.05 absolute accuracy on the primary setting to claim a **specific advantage**. This is expected to fail for the symmetric transfer. It must be reported separately, not rolled into G1.
3. **Cost claim G3:** only claim savings if an implemented downstream adaptive policy beats a fixed-depth policy at matched *measured* evaluations and target accuracy. **Not attempted in this gate**; no model-implied free membrane sampling on a GPU.

No tuning, replacement of the transfer law, seeds, noise, threshold or acceptance criteria after inspecting held-out confirmatory results. Run `python inhibitory_sensitivity_gate.py --output /path/report.json`; unit tests are in `tests/test_inhibitory_sensitivity_gate.py`.

## Controls and interpretive fences
- A repeat measurement, a no-op perturbed channel, and an equally expensive excitatory probe determine whether the result is specific to inhibition or merely generic local derivative sensing.
- The baseline is nonmonotonic *by construction*, so G1 validates a mechanism, not cortical prevalence, novel mathematics or biological plausibility.
- An ordinary derivative estimate is sufficient to explain the result; this is active sensing, not proof of energy-saving compute allocation.
- Better/noiseless current measurement trivially reveals the regime; practical relevance depends on sensing costs and state availability.

## Connection to prior experiments
`GeometricNeuronV24` and `AlternativeNeuron` studied active interventions where passive readouts were ambiguous, and `SihtiBrain` showed identical somatic readouts with different branches. `SihtiBrainV3` was tied by preset probes. This gate isolates a *nonmonotonic branch transfer* and checks an opposite-sign inhibitory response against an **excitation-matched** control rather than claiming new architectural superiority.
