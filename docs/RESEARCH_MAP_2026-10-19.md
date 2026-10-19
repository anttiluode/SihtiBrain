# SihtiBrain — consolidated research map (19 October 2026)

## Why this repository exists

The original question is **not** how to store and retrieve text facts from Qwen. It is whether a structured medium, receiving an input and evolving along local gradients, can make initially hidden distinctions observable when a collective readout sends a *returning signal* through the medium.

Working loop: input → state-dependent local conductance and gradient → dendritic branch state → somatic compression → returning ping → interaction with local branch state → new collective response, optionally followed by a persistent *local* conductance change.

Here 'gradient' has three noninterchangeable meanings:
1. A branch voltage difference creates axial current.
2. A passive reciprocal network relaxes according to an energy gradient under fixed inputs and operator.
3. A **learning** gradient requires an objective and a genuine credit-assignment rule. The first two do not automatically imply the third.

The Sihti residue identity and a transformer residual-sum identity are both useful bookkeeping analogies, **not** evidence that a transformer performs physical energy relaxation or that an activation transplant implements a downward dendritic ping. Hidden-information observability is distinct from spontaneous object segmentation and from useful learning.

## Research hierarchy

### A. Canonical starting point — root of this repository

The root `sihtibrain.py`, demos, `docs/PING_AS_NUDGE.md`, `docs/PING_WITH_STATE.md`, `docs/TUFT_CONTEXT.md`, `docs/REALTIME_CREDIT.md` and associated tests are the merged original SihtiBrain PRs #1 and #2.

- Symmetric seven-compartment tree: clustered versus split leaf input produces the *same entire passive somatic trace* (maximum difference 4.44e−16) while branch voltages and currents differ. A fixed linear somatic ping cannot undo that blindness.
- Nonlinear bistable branch: an identical blind ping can send only the clustered branch into a plateau in the constructed toy. This is **state-dependent readout**, not automatic image segmentation.
- Graded teacher error at the soma can create an EqProp learning gradient. This is **explicitly supervised** and must not be confused with a neutral biological spike.
- Calcium-ping credit and apical context are mechanistic *pathway* demonstrations. In the real-time credit test the simpler three-factor rule matches ping-local credit, with lower single-trial variance; an algorithmic advantage has not been shown.

### B. Main continuation: `studies/sihtibrain_v2/`

Frozen source: SihtiBrainV2 `4a7fa7509c14a0f7941aec992e1652a03d5c7ca3` (stacked PRs #1–#3).

- Initial soma difference, cluster vs split: **0**.
- Fixed-operator linear ping echo: **approximately 0**.
- Contextual state-dependent echo: **−0.1149487**.
- Echo after voltage/context reset and a local slow conductance write: **−0.0481716**.
- When the transient write is removed, the post-reset distinction vanishes.

These are constructed, dimensionless seven-compartment examples, not measured brain data. They show a possible *operator-mediated* way that hidden local differences can alter a later readout.

The tuft-to-basal pathway with a supervised apical residual is a separate gated teaching experiment. Gate 5's preregistered stopping test **FAILED distinctiveness** under shifted distributions: two-ping log calcium ratio scored 62.13% balanced accuracy, while a simpler graded three-factor event scored 62.21% and better BCE (2.664 vs 3.028). Retain the result and its baselines; do not claim superior gradient credit assignment.

Start: [V2 source and tests](../studies/sihtibrain_v2/README.md), [V2 theory](../studies/sihtibrain_v2/docs/THEORY.md), [Gate 5 result](../studies/sihtibrain_v2/docs/GATE5_RESULT.md).

### C. Temporal extension: `studies/sihtibrain_v3/`

Frozen source: SihtibrainV3 `1b5affd88889afc5902dbefe6eb759b967064931` (PR #1).

- Adaptation-driven two-ping probe can reveal matched-soma states in a toy: both adaptive and fixed preset two-phase schemes reached **100%** on the specified noisy cases. The **tie** means adaptation is not necessary to obtain the observability effect.
- In the causal 16-column moving-bar experiment, phase-blind averaging canceled the directional probes. Adaptive/preset/symmetric next-frame top-three recall was **0.615**, versus **0.781** for a simple two-frame velocity forecast. **Tracking advantage failed.**

Read: [V3 Gate 1](../studies/sihtibrain_v3/docs/GATE1.md), [V3 Gate 2](../studies/sihtibrain_v3/docs/GATE2.md).

### D. Side study, not part of the gradient mechanism: `studies/operator_memory/`

These preserve the AIBrain and AIbrainV2 source snapshots, runnable experiments, tests, preregistrations and receipts.

- `aibrain/`: a graph echo collapses to a normalized matrix read in that design. At higher load the equal-size Hebbian memory wins (32 associations: 71.9% graph vs 95.4% Hebbian in the stated noisy condition).
- `aibrain_v2/`: on the preregistered primary memory test, conductance read 55.58% vs 45.41% direct read; a post-result degree-normalized direct approximation reached 53.37%; compact indexed retrieval reached 73.90%. A subsequent TEM-inspired new-route task obtained 82.2% with phase/delta addresses while a smaller indexed coordinate baseline achieved 100%. Later relational-structure and noisy-odometry experiments are retained with original protocols and results.

This is valuable operator-memory research, **not** evidence of a biologically plausible gradient ping. See the original [AIBrain Gate 1](../studies/operator_memory/aibrain/docs/GATE1.md) and [AIbrainV2 gates](../studies/operator_memory/aibrain_v2/README.md).

### E. Archive, not the main path: `studies/transformer/`

- `sihtistack/`: frozen Qwen activation-return gate. On the user's 12-case pretrained Qwen run, no write, correct activation return and wrong-city return all selected the right candidate **2/12**; fact present in context selected it **12/12**. A returned hidden state did **not** restore the specified fact.
- `realtime_readwrite_claude/`: external memory and LLM read/write experiments. Gate 1b: 79.1% two-hop retrieval (predeclared 80% threshold: **fail**). Gate 2 memory-write: 34.0% among eight candidates and 0.3% unrestricted top-1; oracle-write 48.5% among eight and 0% unrestricted top-1. A Gate 2b rehearsal-state protocol exists, but this snapshot does not establish that it passed. These are separate from neuron-gradient flow.

Both are archived to preserve the measurements and negative outcomes. Neither should define the next gradient experiment.

## The next experiment — one coherent direction, not another repository

Use this repository's *state-dependent gradient operator*, not a pretrained LLM. Build a three-level 16→4→2→1 hierarchy with locally retained branch-gradient state and one scalar collective output.

Pre-register:
1. Two different bottom-level configurations yield the same **full first-window top readout** within tolerance while their branch gradients differ. Include out-of-distribution shapes.
2. Send the identical unlabelled return ping through the **actual coupled operators**. Compare fixed linear, locally nonlinear, explicitly state-dependent operator and erased-trace controls. Show both temporal trajectories and scalar readouts; measure discriminatory accuracy on held-out examples.
3. Add a slow, bounded, local conductance write. Erase fast voltages and context, then probe afresh. Require persistent, selective readout change after reset, against equal-state/equal-compute alternative update and normalized/indexed baselines.
4. Report stability and sensitivity to noise, initial symmetry and ping shape. No prespecified semantic answer, teacher-signed error or externally chosen target direction may leak into the unsupervised echo.
5. Only then test *image* object masks against strong elementary image segmentation and tracking baselines. Object separation is currently a hypothesis.

**Stop criterion:** if a normalized linear reader or a fixed two-phase schedule matches the richer mechanism under the same information and compute budget, explicitly record no incremental advantage. Do not retune gate thresholds after viewing the held-out results.

## Reproducibility / what was consolidated

All executable source was copied under nonconflicting paths. The root original remains unchanged except for a navigation banner and this document. Every numerical claim above refers to original toy settings and/or archived measured results, not a new validation by this consolidation process.

This is a **source snapshot** of the named branches, not a Git commit-history merge of unrelated repositories. Old GitHub source URLs may stop working if those repositories are deleted; the necessary code, methods and test files therefore live in this repository. Previous independent runs cannot be re-certified merely by moving their files.

Run core CPU checks, after `python -m pip install -r requirements.txt` and `pip install pytest`:

```bash
python -m pytest -q tests
(cd studies/sihtibrain_v2 && python -m pytest -q tests)
(cd studies/sihtibrain_v3 && python -m pytest -q tests)
(cd studies/operator_memory/aibrain && python -m pytest -q tests/*.py)
(cd studies/operator_memory/aibrain_v2 && python -m pytest -q tests)
```

The Qwen studies require separate substantial model downloads and are deliberately **not** in the standard CPU workflow. The consolidated CI explicitly checks the root and CPU studies. Source scripts and their original work-directory assumptions are preserved; integration into a shared Python package is *not* claimed.
