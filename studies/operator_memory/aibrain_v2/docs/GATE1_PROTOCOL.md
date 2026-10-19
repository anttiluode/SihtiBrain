# AIbrainV2 — Gate 1 precommit

Date: 2026-10-19. This protocol precedes the experiment; outcomes belong in a **separate** results file.

## The question

Does reading memory as the **equilibrium response of a passive cable/conductance graph** preserve associations under interference better than reading **the same stored matrix directly**?

The purpose is to isolate geometry as a readout, not to smuggle in a better writer, more bits, a target-aware probe, or a private addressing oracle.

## Paired storage and task

- Cue: 16 binary dimensions, **exactly four active bits**, distinct random keys.
- Value: one of eight uniformly sampled labels, with no underlying relationship to the cue.
- Storage: 16 x 8 float32 conductances = **512 bytes**; the direct fast-weight baseline stores an identical 16 x 8 float32 matrix.
- Training: present each binding three times with an **identical add-only Hebbian writer** `W += 0.1 * outer(key/4, one_hot(value))` for both memories. Apart from float arithmetic, the two memories therefore contain exactly the same matrix.
- Loads: 8, 16, 32, 64 unique keys; 12 independent seeds (0..11).
- Query: exact cue and held-out one-bit-swap cue (drop one active bit and turn on one inactive bit).
- Rewrite: change 25% of values, write each changed binding three additional times. Query the new targets and the old, untouched associations separately. This is an intentionally **non-erasing** writer, so interference is expected.
- Negative controls: erase stored conductances (must destroy recall); destroy only transitory activity (must not); shuffle input associations (chance-level against original mapping).
- Strong control: exact indexed key/value lookup with nearest-Hamming query, accounting for **3 bytes per binding** (uint16 cue plus uint8 label). This is not an equal-parameter differentiable model; show its actual byte footprint and retrieval accuracy separately.

## Read rules

**Fast matrix:** `score_fast = x @ W`.

**Graph:** build an undirected bipartite network (16 cue nodes + 8 value nodes); edge conductance is W_ij, leak is fixed 0.25 at each node. Drive the 16 cue nodes with x. Let `H(W) z = [x,0]` and score labels by the eight equilibrium values `z[16:]`. Compare the exact solution with iterative Euler relaxation. Because H is a grounded Laplacian, it is symmetric positive definite for every nonnegative W.

Report class-balanced accuracy and raw accuracy (labels drawn uniformly, may have small imbalances), exact vs corrupted, after rewrite both new and untouched. Report paired difference per seed with distribution summary.

## Gate and stopping test

**Operator-read advantage** requires at least **+5 percentage points** in *mean noisy-cue untouched-retention accuracy after rewriting* versus the equal-matrix direct read, without worse than -2 points in noisy-cue new-binding accuracy. These thresholds are intentionally meaningful rather than epsilon-level differences.

If the graph does not pass, retain the negative result and mathematical equivalence/limitations. A more complex update rule would be a new gate, not a retrofit.

## Scope

There is no trained visual object separator, no simulated soma spike, no biological claim, no proof of a new AI architecture. This test asks only whether using geometry as a memory **reader** is useful under controlled interference.
