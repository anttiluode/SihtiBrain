# Gate 2 — relational addressing: preregistered protocol

**Date: 2026-10-19. Commit this before implementing or running the full benchmark.** Subsequent results belong in a different file. This is a *TEM-inspired addressing ablation*, **not** a reproduction of the learned Tolman–Eichenbaum Machine.

## Motivation and epistemic boundary

AIbrainV2 Gate 1 compared unstructured four-hot keys on a fixed matrix. The Tolman–Eichenbaum Machine (Whittington et al., *Cell*, 2020) instead factors a reusable structural state `g` from environment-dependent observations `x` and binds the two in fast memory. Our first test provides, rather than learns, the transition algebra. We must **not claim emergence of grid cells, learned graph structure, hippocampal dynamics or superiority over a classical indexed graph**.

## The world and observation constraints

- A 6×6 toroidal grid, actions N/S/E/W with fixed group structure `Z_6 × Z_6`. A world changes node sensory symbols and value labels, **not** action meanings.
- Each world independently assigns one of **four aliased sensory symbols** and one of **eight arbitrary target values** to each node, without correlating the two.
- An agent begins at the anchor with structural state `g(0,0)`. Train on **18 or 30 distinct locations** per world, including the anchor; 3 identical presentations per association.
- At each training observation, the agent gets a sequence of actions from the anchor and the resulting sensory symbol/value. The **target node ID and integer coordinates are never supplied to the phase-memory writer**.
- Tests query already experienced locations using **a different action order and inserted inverse-action detours**. The sensory symbol is visible, but the value to retrieve is hidden. No memory reads its target from the query.
- Distinguish **revisited by an unexperienced path** from **unvisited location**; unpredictable random target labels at truly unvisited locations cannot be inferred by any method without additional prior information. Explicitly test this impossibility control.

## Readers and writers

1. **Grid-phase + delta**: action-conditioned 12-dimensional sin/cos state, with three harmonics per axis. Coordinate updates are rotations, no node-ID oracle. Bind a 1-of-8 target to the final address with an error-correcting delta fast weight; 8×12 float32 = **384 bytes**.
2. **Grid-phase + Hebb**: the same phase codes and storage, but an additive Hebbian writer, testing whether delta updates matter.
3. **Content + delta**: same 384-byte matrix and delta update, with cue generated only from one of four sensory symbols, not action history. Padding/normalization cannot create missing location information.
4. **Time + delta**: same 384-byte matrix, address generated from absolute action count, not relational location (so detour length matters). Report collision/alias effects.
5. **Grid-phase + direct indexed values**: phase or coordinate from actions plus explicit table of `coordinate -> value`. The table is a **strong structured baseline**, not an oracle ID: it derives coordinates from exactly the same action sequence. Count bytes (two uint8 coordinates and one uint8 label per association). It should be allowed to overwrite a value exactly.
6. **Graph traversal + indexed values**: build an explicit action transition graph from observations, and permit traversing edges only if known; additionally compare the same **known torus transition prior** used by the phase model. The prior-enabled classical method should tie or exceed learned associative recall; report prior separately from observed-edge-only version.
7. **Phase address with shuffled actions**: negative control, deliberately wrong action semantics.

Keep all readout weights/frequencies/learning rates fixed for the entire run. No validation test tuning. Comparisons 1–4 are all **384-byte** fast-weight matrices; indexed alternatives have a different but explicitly reported byte budget.

## Worlds and metrics

- 20 independent, held-out worlds, seeds 1000..1019; each new world resets episodic fast memory and draws fresh sensory symbols and labels. Structural coordinate encoder is shared unchanged.
- For each visited-count `18,30`, report:
  - exact/new-path recall at visited nodes (one query per visited node),
  - recall after changing 25% of experienced associations (changed and untouched separately), with **3 further writes per change**,
  - one-query-at-each-unvisited-node performance (random-label floor 12.5% expected in aggregate),
  - path-address consistency, including a closed-loop test,
  - byte footprint and operation counts or clear complexity bounds.
- Average across 20 worlds and report per-world paired differences and uncertainty. Keep separate the previously seen location/new-route test from genuine never-seen sensory-label prediction.
- Negative controls: zero memory; shuffled labels; wrong actions; overwrite old target at a known coordinate.

## Success / stop

**Gate 0 (necessary)**: phase address of the same location is stable to <=1e-9 Euclidean error along alternative routes and closed loops in 100% of the tested sequences, and distinct node codes do not collide under nearest-code retrieval.

**Gate 1 (addressing repair)**: phase-delta improves **>=20 percentage points** in visited-location/new-route accuracy over both content-only and time-only *equal-byte delta* controls at visited count 18, while exceeding chance by >=30 points at both loads.

**Claim of a new algorithmic advantage** needs a different condition: phase-delta must outperform the **structured indexed graph/coordinate baseline** on accuracy at comparable memory and compute cost. We expect that harder gate to fail in random-value torus worlds. If it fails, **retain only the demonstration that a regenerable structural address fixes the artificial aliasing problem**.

**Gate 2 (rewrite)**: report changed and untouched accuracy after 25% rewrites, whether or not a gate passes. Avoid presenting a failed update as a property of learned cortical credit.

The *learned address* question requires a future separate gate that does not hand the model the action algebra. Only then consider RealTimeReadWrite or a frozen LLM.

Reference: Whittington et al. (2020), *The Tolman–Eichenbaum Machine*, Cell 183, 1249–1263, doi:10.1016/j.cell.2020.10.024.
