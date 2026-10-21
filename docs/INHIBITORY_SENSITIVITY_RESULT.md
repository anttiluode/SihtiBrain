# Inhibitory Sensitivity Gate — results (2026-10-21)

**Protocol frozen first:** [INHIBITORY_SENSITIVITY_PREREG.md](INHIBITORY_SENSITIVITY_PREREG.md) — committed before the executable source and before confirmatory runs.

## Verdict

- **G1 / locally observable hidden regime: PASS (in independent numerical reproduction).** The smooth transfer has two distinct currents with **identical baseline response**. A small inhibitory perturbation makes one response fall and the other rise. At the prespecified primary noise level, the sign-only reader achieves **99.94%** (3,198/3,200) while an equally costly no-op two-read control gets **51.09%**.
- **G2 / specifically inhibitory benefit: FAIL.** An equally costly *excitatory* perturbation gets **100%** at primary noise. The distinctive-inhibition gate demanded a gain of at least 5 percentage points; inhibition has **−0.06 points**. It does not win.
- **G3 / adaptive-computation savings: NOT TESTED.** Every perturbative arm pays **two activation evaluations and two measurements**, compared with one passive evaluation. This experiment has no local planner or measured compute advantage.

These figures were independently reproduced using the source algorithm with the frozen seeds. The checked-in Python experiment, \`inhibitory_sensitivity_gate.py\`, creates a full machine-readable, per-seed receipt via \`--output\`; GitHub Actions runs its regression suite. This toy's algebra and noise-distribution symmetry were chosen deliberately. It does not establish a new biological or ML principle.

## Confirmatory results

Four seeds (41, 97, 127, 211), 800 *balanced* examples/seed, 3,200 examples per noise condition. No fitting or tuning of thresholds on confirmatory trials.

| RMS scalar-read noise | Inhibitory accuracy | Excitatory accuracy | Sham two-read accuracy | Passive accuracy |
| --- | ---: | ---: | ---: | ---: |
| 0.015 (primary) | **99.94%** | **100.00%** | 51.09% | 49.72% |
| 0.040 | 91.41% | 91.63% | 51.09% | 49.84% |
| 0.080 | 75.81% | 74.78% | 51.09% | 49.34% |

Per-seed primary inhibitory accuracies: **99.875%, 99.875%, 100%, 100%**. Per-seed primary excitation: **100%, 100%, 100%, 100%**. Every trial is paired in hidden-input distance from the peak with a trial of the other class.

At zero noise, both perturbation directions decode perfectly. With independent noise on each read, the effective differential noise standard deviation is \`sqrt(2) × sigma\`. The performance drop at higher noise is therefore expected. Nothing in the test measures synaptic plasticity or task error.

## Why the result works (and what it fails to establish)

We *engineered* \`I=1-u\` and \`I=1+u\` to have the same output under \`A(I)=exp(-(I-1)^2)\`. A change in output under a small perturbation supplies local slope information that the baseline scalar lacks. Overdrive can be “rescued” by inhibition **in this specified transfer curve**; the opposite effect appears for the weak state.

An excitatory probe observes the opposite slope just as cheaply. This means a generic, locally available susceptibility signal could be an extra sense, but it does **not** make inhibitory gating unique or justify the assertion that a membrane supplies that susceptibility free of physical cost.

There is no evidence here that a neuron must dynamically allocate additional layers or that an artificial neural network can avoid a nonlinear function evaluation. The previously observed \`SihtiBrainV3\` preset-probe tie, \`AlternativeNeuron\` silent-switch limit and \`GeometricNeuronV24\` expected-information probe cost remain important prior controls.

## Running the source and preserving the receipt

From the repository root:

\`\`\`bash
python -m unittest discover -s tests -p 'test_inhibitory_sensitivity_gate.py' -v
python inhibitory_sensitivity_gate.py --output results/inhibitory_sensitivity_gate.json
\`\`\`

The result file is **generated**, not manually edited. It records all condition accuracies, per-seed outcomes, exact-check diagnostics, cost accounting and the three gate verdicts.

**Next gate, not performed here:** If inhibition changes an overdriven state so that a *future needed computation* becomes possible while a matched excitatory or simply repeated evaluation cannot, pre-register that practical distinction on a stateful system, with energy/latency costs. Do not claim it from this memoryless response curve.
