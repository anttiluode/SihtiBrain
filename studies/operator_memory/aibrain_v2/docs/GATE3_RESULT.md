# Gate 3 — learned action structure, held-out transfer and memory limits

**2026-10-19.** [Gate 3 protocol](GATE3_PROTOCOL.md) was committed before implementing the full benchmark. Source observations and all random seeds are fixed. This is **symbolic finite-state system identification plus a spectral graph address**, not learned TEM or neuroscience.

## The structural learner

Eight source environments randomly relabel every state with opaque, uniquely identifiable landmark strings. The learner is given **2,400 action-labeled landmark transitions per source world** and a reset anchor, but no state coordinates, torus modulus or node IDs. It reconstructs a transition table, assigns canonical states via BFS in declared action order, and builds a 12D code from the twelve lowest nonconstant graph-Laplacian eigenvectors.

**Gate 0 PASS.** The eight separately permuted worlds yield an identical canonical **36-state / 144-directed-edge** automaton. Two distinct routes to the same destination agree in 36/36 cases, the four-action closed loop returns to the anchor, and the minimum distinct spectral-key distance is ~1.00. Shuffling action labels in the training data makes transitions inconsistent and is correctly rejected.

**Important constraint:** source landmarks are unique, making system identification easy for a classical learner. Held-out test worlds have only four aliased sensory symbols. The world generator is a hidden 6×6 torus, and the same action vocabulary and transition structure recur in every world. The learner does *not* discover that shared structural assumption; it is the experimental setup. There is no slow neural parameter fitting, no recurrent learned RNN grid state, and no full Tolman–Eichenbaum Machine implementation.

## First held-out result (20 new worlds)

| Method | New route / 18 visited | New route / 30 visited | Changed bindings / 18 | Untouched retention / 18 | Total stored bytes at 18 |
|---|---:|---:|---:|---:|---:|
| **Learned graph + spectral delta** | **83.1%** | 54.5% | **100%** | 64.6% | **2256** |
| Learned graph + random-code delta (fixed seed 42) | 77.8% | 56.3% | 100% | 62.5% | 2256 |
| Learned graph + spectral Hebb | 85.6% | 68.5% | 53.8% | 82.9% | 2256 |
| Content-only delta | 34.2% | 21.0% | 72.5% | 19.6% | 384 |
| Time-only delta | 18.3% | 16.0% | 7.5% | 15.7% | 384 |
| **Learned graph + indexed table** | **100%** | **100%** | **100%** | **100%** | **162** |
| Target-world observed edges only | 0% | 0% | 0% | 0% | variable |
| No memory | 10.8% | 11.0% | 13.8% | 12.1% | 0 |

The classical learned-graph indexed model receives **exactly the same learned action transition table and anchor as the spectral model**. It does not receive target-world node IDs. It just has a better random-value storage mechanism. The observed-edge model is a deliberately restricted control and fails because detours use edges unseen in its target-world recordings.

**Gate 1 PASS**: learning the structure makes unseen-route recall much better than content or time-only keys with equal readout-matrix bytes. The comparison is not equal total storage once the structural table and spectral code are charged.

**Gate 2 borderline PASS, preregistered single-seed comparison**: spectral scores 83.06%, one seeded random graph-state code scores 77.78%, a **5.28-point** advantage over the fixed seed. At 30 visited states random codes slightly beat spectral (56.33% vs 54.50%). A separate *post-result* multi-seed random-code sensitivity check was therefore added, and its results must not be misrepresented as preregistered.

**Hard superiority gate FAIL**: the structured index beats spectral recall with dramatically less storage.

## Post-result 16-seed random-code sensitivity (not preregistered)

After observing the original 5.28-point advantage against one frozen random-code seed, we reran the exact same learned graph and twenty target worlds with **16 new random graph-state code tables** (seeds 42000..42015). This is deliberately a *post-result* diagnostic rather than a redefinition of the original decision.

| Visited locations | Spectral delta | Mean random-state delta | Spectral minus mean random | Random-seed wins by spectral |
|---|---:|---:|---:|---:|
| 18 | **83.06%** | 79.77% | +3.28 percentage points | 14/16 |
| 30 | 54.50% | **56.05%** | −1.55 percentage points | 3/16 |

At 18 observed states, the result is suggestive that graph-spectral structure is useful beyond an arbitrary state code, **but the advantage does not reach the originally set 5-point threshold against the mean of these new random seeds**. At higher load it reverses. Neither result supports the claim that spectral structure is a new efficient memory system: graph indexing remains perfect, much smaller and simpler.

The extra code seeds are *not* independent world datasets; all were compared on the same twenty worlds. Do not count them as 320 independent worlds or treat this as a preregistered significance test.

## Storage costs and the stability–plasticity limit

The spectral/delta approach needs 144 bytes for the learned transition table, 36×12×4 = 1728 bytes for the spectral key table, and 8×12×4 = 384 bytes for its read/write matrix: **2256 bytes total**. The indexed approach retains the **same learned 144-byte structural table**, plus 18 target binding bytes = **162 bytes** (174 bytes for 30 bindings).

At 30 visited nodes, spectral/delta has 100% changed-label recall but retains only **28.9%** of untouched associations; spectral/Hebb has 43.6% changed recall but retains 63.7% untouched. Neither addresses interference as well as an explicitly indexed map.

Random target labels at **never-visited nodes are not inferable**; the correct indexed method returns "unknown", scored as 0% under forced-label exact accuracy, and should not be penalized for calibrated abstention. This task measures *returning via an unseen route to a seen node*, not prediction of a random unseen label.

## What now?

Gate 3 establishes that an action transition graph **can be learned from distinctive experiences** and transferred to novel worlds while retaining consistent addresses. It does **not** establish a new AI learning algorithm. A classical learned automaton is both more accurate and smaller in this regime.

The next meaningful discriminator is **learn the structural state from aliased, partial, noisy trajectories**, where a single sensory token cannot identify a source-world state. That requires a fair baseline with learned symbolic automata, Bayes filters, recurrent state models, and explicit graph indexing. It should be a separately preregistered Gate 4, not a retrospective claim that the source observations already solved this difficult problem.

## Reproduction

    python -m unittest discover -s tests -v
    python demo_learned.py --output runs/learned_gate3
    python diagnostic_gate3_random_seeds.py --output runs/learned_gate3

Run results, a four-panel visualization, complete per-world JSON and CSV, plus post-result random-code-seed diagnostics are uploaded by GitHub Actions on Python 3.11 and 3.13.

Relevant prior work: Whittington et al. (2020), *The Tolman–Eichenbaum Machine*, doi:10.1016/j.cell.2020.10.024.