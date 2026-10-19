# AIBrain: queryable connection memory

## Intent
Translate the SihtiBrain line into an executable AI memory block on a CPU.
The user requested implementation and measurement in this new repository.
The first experiment asks whether new associations can be written online,
recovered by partial queries, and retained after transient activity is cleared.
It does not attempt an entire cognitive architecture or image segmentation.

## Alternatives and choice
1. **Bounded associative connection memory (chosen):** compact, measurable,
   no backpropagation or pretrained network; its relationship to familiar
   fast-weight memory can be derived exactly.
2. Full graph segmentation/tracking: closer to the initial visual analogy,
   but combines grouping, correspondence and plasticity before any is isolated.
3. Transformer integration: worthwhile only after measuring the primitive;
   large training runs obscure the mechanism and exceed this gate's scope.

## Block
Input is a sparse binary key x in R^64 and a co-observed one-hot value v in R^8.
A Sihti-inspired **common-mode split**, not the original spatial Sihti sieve,
keeps core=mean(x) and residue=x-core. Reconstruction is exact before query
normalization. The pooled scalar core is identical for all eight-active-bit keys.

Persistent state: W in [0,1]^(64x8), float32. On observing (x,v), each active
row learns a co-occurrence: W_ij <- (1-eta*x_i) W_ij + eta*x_i*v_j, eta=0.5.
This is a local exponential association writer, not a behavioral loss gradient.
Observed values explicitly supply the association: it is paired/self-supervised
storage, not discovery of semantic categories without information.

Query q is the unit-normalized residue of a possibly corrupted cue.
Clamp q at key nodes of a bipartite positive-conductance graph. Value nodes h
relax under E(h)=leak/2 sum_j h_j^2 + 1/2 sum_ij W_ij(h_j-q_i)^2.
Then h*=W.T q / (leak + column_sum(W)), leak=1.
The eight-dimensional echo is the readout; argmax predicts the value.
This is a query-dependent response, not literally a scalar biological spike.
Its exact direct normalized matrix-read equivalent must be reported alongside it.
Iterative relaxation must converge to the direct answer and decrease E.
Queries must not update W. Transient h can be cleared without erasing W.

## Evaluation fixed before first benchmark
Five seeds: 41,97,127,211,307; 20 independent episodes per seed.
Loads: 8,16,32,64,128 fresh keys per episode; 8 values, balanced assignments.
Cue corruption swaps 0,1,3 active/inactive bits, preserving the pooled scalar.
Each association observed once, with all methods seeing the same order.
No hyperparameter tuning on these evaluation seeds.

Controls: pooled read, erased W, frozen writer, shuffled queries, raw (uncentered)
cue, direct normalized read (must tie), equal-matrix-size Hebbian and delta-rule
fast weights, and full-history softmax attention. Report persistent float32 array
bytes; history attention has a load-dependent budget and is not a matched-size
competitor. Report latency separately from accuracy, with local-machine caveats.

Rewrite gate at load 32: change first quarter of associations by one value,
observe changed pairs once and four times (two separate branches from the original
memory), report changed recall and untouched recall separately. Cache attention
stores repeated events and may suffer obsolete bindings; no recency oracle.

Mechanism gates: reconstruct split, matched scalar core, recall after clearing h,
loss after clearing W, frozen-writer floor, direct/iterative equivalence.
No mandatory performance-win threshold. Negative results remain in the report.

## Deliverables
Python 3.10+ package, NumPy core; Matplotlib figures; unittest suite; benchmark CLI
with JSON receipts and SVG figure; a standalone browser demo with editable cues,
write, query, clear activity and erase memory; README and measured Gate 1 report.
No GPU, external datasets, network calls, pretrained weights or hosted service.
Invalid dimensions, NaN/Inf, invalid write ranges, invalid CLI loads/episodes and
constant/empty cues must have documented behavior.

## Interpretation and scope
This can establish online associative storage and query-conditioned readout.
If the matrix equivalent ties, relaxation itself supplies no extra information.
Capacity, overwrite interference, storage budget and runtime must be visible.
No claim about brain function, J-space identity, autonomous object basins,
novelty over fast weights, or superiority over transformers follows.

## Reference snapshots inspected
- SihtiBrain, claude/ping-as-nudge at afc1e54: local hidden state and supervised feedback.
- SihtiBrainV2, codex/gate5-credit-stopping-test at 4a7fa75: persistent writes;
  stopping test failed to show a distinctive two-ping advantage.
- SihtiBrainV3, codex/v3-alternating-probe at 1b5affd: two-phase observability;
  phase-averaged tracker tied symmetric routing and lost to velocity forecast.
The supplied SihtiBrainV URL did not resolve; SihtiBrain is the reference from
context. All reference repositories remain read-only.
