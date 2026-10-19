# Gate 4 pre-registration — sparse place expansion and noisy action correction

**2026-10-19. Precommitted before executing the new Gate 4 benchmark.** Builds on the learned graph and spectral addresses of Gate 3. The expansion is a nonlinear feature map of an already distinct 12-dimensional code, **not the recovery of irretrievably discarded information**. Gate 3 and the earlier SihtiBrain repos remain untouched.

## Hypotheses

H1: Under identical writer, world and update schedule, a fixed random-projection + ReLU + top-k *place-cell-style* expansion sharpens code selectivity and improves held-out **new-route recall** and **untouched retention** over the 12D spectral-key fast-weight reader. The top-k projection is random and frozen. No label information enters the key.

H2: When odometry reports occasional wrong actions, a state estimate corrected by observed (aliased) sensory symbols can outperform open-loop action integration. **This is a filter/measurement test, not a claim that continuous phase is inherently more accurate than integer coordinates.** Give a Bayesian indexed graph the *same* action stream, learned graph, noise model and sensory observations.

## Source and target worlds

Retain Gate 3's exactly observed eight source landmark worlds (3000–3007), reconstructed canonical 36-state/144-edge graph from 2400 action-labelled transitions each, and the frozen learned 12-dimensional Laplacian spectral key. The source graph is reconstructed by a deterministic *symbolic* learner; this is not end-to-end TEM.

Twenty untouched target worlds (1000–1019), either 18 or 30 of 36 randomly labelled nodes visited and written three times; arbitrary independent 8-way values and four aliased sensory categories. Query only previously seen locations by alternative action paths with inverse-action detours. After the first test, rewrite 25% of bindings three times; report changed and untouched separately. Never-visited random values are not inferable.

## Expanded readers

- **Spectral-12 delta**: Gate 3's exact reference, 8×12 float32 weights.
- **Expanded-64 sparse delta** and **expanded-256 sparse delta**: seeded random Gaussian projection R of size D×12, ReLU, keep only the top-k strictly positive activities (k=8 for 64; k=16 for 256), then L2-normalize. Same delta write formula `W += .75 (onehot(value)-W@key) outer key`. Frozen projection seed 41 across all worlds. Count the projection matrix bytes, learned 36×12 spectral code table, 144-byte transition table and fast-weight matrix, including when the projection could theoretically be deterministically regenerated from a seed.
- **Expanded-256 dense ReLU delta**: same R, keep all positive activities; tests whether sparsity rather than just dimensional lifting matters.
- **Random 256D sparse ID codebook**: a fixed independent random unit/top-k code for each learned graph state. Counts all 36×256 codebook bytes, graph and fast weights. This is the high-capacity address baseline that shows whether learning a graph-based expansion is superior to a keyed random hash.
- **One-hot learned-state delta**: 36D one-hot address reconstructed from the same learned graph. Count transition table + 8×36 float32 weights. This is a full-band, explicit-address associative read, not an independent learned representation.
- **Indexed graph**: same learned 144-byte transition table plus one uint8 value per written node. Must remain the strong baseline.

The sparse place-cell claim only passes if the **256D expansion** beats the 12D delta baseline by >=10 percentage points in both new-route recall AND untouched retention at **30 visited locations**, while keeping changed-binding recall >=95%, and beats the 256D dense ReLU control on at least one primary metric. More dimensionality and stored bytes are always disclosed; any win is **not** an efficiency advantage. Do not retune top-k, projection seed or hyperparameters after seeing the result.

## Noise and correction

Generate an actual alternative route for each already-visited target node. For each physical step, the logger substitutes a uniformly chosen *wrong* action with probability 0%, 15%, or 30%; the actual movement is unchanged. The receiver sees **only reported actions**, and a noisy sensory token emitted by the *true current node* after each step (5% sensor substitution into one of the other three symbols). Target-world mapping from graph states to sensory symbols is known **only at nodes whose labels were observed during training**; all other nodes are unknown. The agent never receives the target coordinate, actual route, or target label at query time.

A 36-state Bayesian filter uses the learned transition table and the declared action/sensor error probabilities. Unknown-state emissions are uninformative. It starts at the known anchor and updates on every reported action/observed symbol. Compare:
- Open-loop corrupted-path learned graph + indexed lookup;
- Open-loop corrupted-path expanded sparse memory;
- Bayes-filter MAP address + expanded sparse memory;
- **Bayes-filter MAP address + indexed lookup**, same available sensors;
- **Bayes-filter full value posterior + indexed lookup** (stronger standard decision rule). This can abstain if no indexed state is sufficiently likely; for classification results report forced-label accuracy and unknown separately.
- Oracle true-route indexed lookup **only as an upper bound**, clearly marked not a deployable model.

Correction helps if averaged across 20 worlds and loads, at 15% action error, MAP-address accuracy gains >=5 percentage points over the matching open-loop baseline **and** posterior-index lookup is not beaten by an information-rich candidate that receives the same observations. As a separate calibration metric report posterior true-state probability/rank, with no state knowledge used for the inference.

## Accounting and limits

The 256D sparse read has at least 144+1728+12288+8192 = **22,352 bytes** of persistent stored data. The index needs **144 + visited_count** bytes for keys/values; for sensory correction add one byte per visited node in both. Matrix and index operations/solver state are different; report them, do not claim equal resources.

All results are deterministic synthetic experiments with fixed seeds. This is not a reproduction of hippocampal place cells or learned TEM, not automatic object separation, and not proof that Fourier information was removed and biologically regenerated. A deterministic map of a lossy code cannot reconstruct genuinely aliased inputs. The tested 12D code distinguishes all 36 locations already; expansion can change conditioning and crosstalk, not add missing mutual information.

If the expansion fails, preserve the negative result. If sensory correction helps both methods, attribute it to measurement fusion rather than a proprietary phase mechanism.
