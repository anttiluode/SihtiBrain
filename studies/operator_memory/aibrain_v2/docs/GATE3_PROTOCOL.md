# Gate 3 preregistration — learn and transfer the action automaton

**2026-10-19. Precommitted before experiment implementation or final run.**
This follows [Gate 2](GATE2_PROTOCOL.md) and is stacked on its PR branch, not merged into main.

## What changes

Gate 2 was given the \`Z6 × Z6\` torus action algebra. Gate 3 does **not** supply that graph or coordinates to the candidate: a learner receives action-labeled sequences of *distinct sensory landmarks* in eight source worlds, infers each world's transition automaton, and constructs a **canonical action-ordered graph** rooted at a designated reset landmark. Different landmark permutations must produce the same learned automaton. Source-world observations are only (landmark_before, action, landmark_after); there are no integer coordinates, modular arithmetic instructions or target values. There is no neural training in this version: it is explicit symbolic transition-system identification plus an unsupervised graph-spectral address (a classical method). We must not call this a trained TEM.

Held-out target worlds retain the same action-transition structure but replace source landmarks with **four aliased sensory symbols** and randomly assigned eight-way target values. These are the same kind of independent labels as in Gate 2. The learned structure, not an oracle torus coordinate, provides navigation.

## Fixed train/test schedule

- Hidden generator for evaluation: 6×6 wrapped grid with N/S/E/W action vocabulary, 36 latent states. World landmark IDs are randomized unique strings in source worlds and unrelated to latent coordinates. The learner is not told the grid size or transition formula.
- Pretrain on **8 random landmark worlds** (seeds 3000..3007). Source observation collection performs **2400 random-walk transitions per world** from a designated anchor, then explicitly diagnoses graph completeness. If any world leaves edges missing, report fail; do not silently fill transitions from the simulator.
- Canonicalization: from the observed source-world transition table, BFS outward from the known reset landmark, traversing action names in a fixed declared order. The resulting state IDs are **learned graph-relative addresses**, not environment coordinates.
- Transfer invariants: all eight independently permuted source worlds must recover *identical canonical action tables*; a closed loop and two distinct action routes to the same source state must agree. Check correctness using transitions **held out from the learner's data representation**, by reconstructing independent routes after training, never supplying coordinates.
- Learn a structural **12-dimensional spectral code** from the discovered *undirected* graph Laplacian (lowest twelve nonconstant eigenvectors; normalize each state's row), with deterministic eigensolver, no target-world labels. The code table is frozen across target worlds.
- Control address: independent seeded random 12-dimensional unit vectors indexed by the **same learned graph**. This tests whether graph spectral geometry contributes beyond simply using correct symbolic addresses.
- Target worlds are the predeclared Gate 2 20 fresh worlds (1000..1019), 18 or 30 observed associations per world, three presentations each. Revisit through new action orders and inverse-action detours, rewrite 25% of values, measure changed and untouched recall, and query never-visited nodes separately. Reuse the *same* generated worlds for all arms.

## Methods

1. Learned graph + spectral address + error-correcting delta memory, **8×12 float32 (384 bytes) for associative weights**, plus the stored structure and code table.
2. Learned graph + random-state address + identical delta memory and size; ablation.
3. Learned graph + spectral address + additive Hebb memory (same weights/storage size).
4. Content-only 12D delta, as in Gate 2.
5. Action-count-only 12D delta, as in Gate 2.
6. **Learned graph + explicit indexed table**, with the *same transition table and anchor* as the spectral candidate; precise exact recall and rewrite reference. Count graph and bindings memory separately.
7. Observed-edge-only per-target-world graph, no structural transfer; should fail at routes containing missing edges.
8. Shuffled action labels or broken training transitions: negative control, should reduce accuracy or graph consistency.

The target-world learner sees only actions, sensory symbols, observed values, and the anchor reset. It does not get latent coordinates. No target value is shown during read.

**All memory accounting must include the learned action table (36 states × 4 transitions; compact uint8 table = 144 bytes), spectral code table (36×12 float32 = 1728 bytes), and 384-byte fast weights. A compact learned graph+indexed values uses the same action table plus one uint8 binding per visited node.** Do not present the phase/delta scheme as merely 384 bytes total.

## Gates

**Gate 0 (necessary):** all eight source worlds' canonical transition tables agree, have 36 states and 144 complete outgoing edges, and distinct routes to each destination give identical addresses; spectral rows are unique (minimum pairwise distance > 1e-6).

**Gate 1 (addressing):** on 18 visited nodes, spectral/delta recalled target label after a novel route at least **20 percentage points** better than each content-only and step-count delta baseline, with mean accuracy >= 0.45 over 20 worlds. At 30 visited nodes still >chance by >=20 points. Report seed-wise paired differences.

**Gate 2 (spectral geometry added value):** spectral/delta improves new-route recall by at least **5 points** over the same learned-graph random-state-address delta, at 18 visited nodes. Report this even if it fails.

**Hard superiority gate:** spectral fast-weight storage/retrieval must beat the learned graph + indexed values in accuracy *and* total memory footprint. On random arbitrary target values, this is expected to fail. A positive addressing gate is not new algorithmic superiority.

If Gate 0 fails, stop; do not interpret later accuracy.

## Limits and stopping conditions

The learner is given **unique landmark identities and a reset anchor in source worlds**, allowing conventional exact finite-state system identification. This is not the harder problem of learning a state model from heavily aliased observations, nor a differentiable TEM. The target world's action structure is assumed identical to the learned source graph, making transfer possible. We should neither suggest the model discovered 2D geometry from first principles nor that this outperforms symbolic planners. A future next gate may learn transitions from aliased noisy observations without landmark IDs, with equal-protocol symbolic, contrastive, and recurrent baselines.

Reference: Whittington et al., The Tolman–Eichenbaum Machine, Cell 183 (2020), doi:10.1016/j.cell.2020.10.024.