# Gate 1 — useful association memory, no distinctive ping advantage

The positive result is small and concrete: an online connection matrix stores
new arbitrary associations, a corrupted cue retrieves them, and clearing
value-node activity leaves retrieval intact. The negative result is equally
concrete: this bounded row-averaging writer performs worse than ordinary
Hebbian or delta-rule memory of the same size at larger loads.

## Protocol

Design and protocol were committed before the first full benchmark. Defaults:
seeds 41,97,127,211,307; 20 independent episodes per seed; 8,16,32,64,128 pairs.
Each episode has fresh binary 64-dimensional keys, exactly eight active features,
and balanced, randomly assigned values among eight outputs. Key generation
rejects duplicates. No query target reaches a memory's read function.
Each model sees exactly the same write stream, once. No parameters are tuned
on evaluation seeds. There is no pretrained feature extractor or decoder.

Cue corruption swaps 0,1,3 active/inactive bits per query. This changes exactly
0,2,6 pixel bits while preserving the mean at 0.125. Every stored key is queried
under every corruption condition. Each row in the JSON records the seed, load,
method, mean accuracy and all 20 episode accuracies. The plot shows means and
sample standard deviation over the five seed means, not confidence intervals
over independent query trials.

## Architecture

Persistent W is a 64×8 float32 matrix, initially zero. At a write, rows move
locally toward the observed value: `W += 0.5*x[:,None]*(v[None,:]-W)`.
This preserves conductances in [0,1]. It also overwrites other associations
that share those rows. The observed value is information supplied by the
experience; absence of backpropagation does not remove that supervision.

The key is split exactly into its scalar mean and zero-mean residue. The
normalized residue is the signed query q. These positive and negative drives
are clamped at the key nodes, while the eight value nodes descend

```math
E(h)=\frac12\|h\|^2+\frac12\sum_{ij}W_{ij}(h_j-q_i)^2.
```

The unique equilibrium is `h=(W.T@q)/(1+W.sum(axis=0))`. The transient activity
has eight dimensions; the query has 64. This is no longer the symmetric scalar
soma-pulse experiment. Known partial keys supply the address information.
The graph has a unique convex equilibrium, not spontaneously discovered object
basins. The energy descent implementation uses 80 preconditioned steps; the
benchmark uses the mathematically equivalent direct solution.

## Input behavior

Write keys must be finite vectors in [0,1] of the configured dimension. Values
must be nonnegative probability vectors; sums within absolute tolerance 1e-8
are normalized to one so repeated writes cannot exceed the conductance bound.
Queries accept finite signed cues and scale before normalization to avoid overflow.
Constant centered cues yield zero scores. The standalone split rejects a residue
whose range cannot be represented by finite float64 numbers.

## Comparison rules

| Method | Write/read rule | Persistent numeric state |
|---|---|---:|
| Connection | Local bounded row EMA; degree-normalized echo | 2,048 B |
| Direct equivalent | Same W and exact normalized matrix read | Same W |
| Hebbian | Add `outer(centered_unit_key, value)`; linear read | 2,048 B |
| Delta | Add `outer(key, value − predicted_value)`; unit rate | 2,048 B |
| Attention | Store every normalized key and value; cosine softmax, temperature 12 | 288 B/event |
| Raw query | Same connection memory, query without centering | Same W |
| Shuffled | Same connection memory, permuted query-to-target correspondence | Same W |
| Erased | Trained connection matrix zeroed before evaluation | 2,048 B |
| Frozen | Zero-initialized matrix, never written | 2,048 B |
| Pooled | Analytic constant-output control: equal scalar key mean, balanced labels | No binding information |

Every method outputs the same eight-dimensional value scores. Argmax ties
choose the lowest index. Balanced labels make uninformative zero scores yield
12.5%. The attention cache is not a matched-byte comparator except approximately
at the smallest load; it keeps the full history. It has no key deduplication or
recency policy, and is intentionally an append-only event cache in the rewrite
test. A cache with explicit latest-binding updates is a separate stronger
rewrite baseline. The baseline learning rates are specified in advance, not
optimized; this is a gate for this implementation, not a general ranking of
memory families.

## Recall with one active/inactive bit swap

| Pairs | Connection | Hebbian | Delta | Attention | Chance |
|---:|---:|---:|---:|---:|---:|
| 8 | 100.00% | 100.00% | 100.00% | 100.00% | 12.50% |
| 16 | 95.94% | 99.81% | 99.25% | 100.00% | 12.50% |
| 32 | 71.91% | 95.41% | 92.03% | 100.00% | 12.50% |
| 64 | 45.19% | 80.31% | 70.33% | 100.00% | 12.50% |
| 128 | 29.04% | 59.45% | 46.00% | 100.00% | 12.50% |

With clean cues, connection recall is 100%,97.44%,77.66%,48.45%,30.58% over
these loads. With three swaps it is 99.12%,84.06%,58.34%,36.92%,24.59%.
At load 32 with one swap, raw queries reach 71.47%; shuffled queries reach
11.44%; erased/frozen/pooled reach 12.5%. Centering is therefore not an
established source of useful advantage here.

## Rewrite gate

Use new independent load-32 episodes. Change the first quarter of the bindings
by adding one modulo eight to their values. Keep the remaining truth untouched.
Start each of the one-exposure and four-exposure conditions from a separate
copy of the original trained state. Evaluate all methods with clean keys.

| Method | Changed, 1 exposure | Retained, 1 exposure | Changed, 4 exposures | Retained, 4 exposures |
|---|---:|---:|---:|---:|
| Connection | 98.63% | 69.75% | 99.38% | 42.46% |
| Delta | 99.50% | 92.79% | 100.00% | 91.33% |
| Hebbian | 54.25% | 97.21% | 100.00% | 63.17% |
| Append-only attention | 61.88% | 100.00% | 100.00% | 100.00% |

The connection writer favors recent observations; overlapping key features
lose old bindings. Repeating a write is not selective repair. The simpler
delta rule gives better changed/retained behavior in this gate.

## Mechanism checks and runtime

- Exact mean-plus-residue reconstruction error: 0.
- Equal scalar core across all episode keys: exact.
- Clearing activity preserves the echo; queries preserve connections.
- Erasing connections and freezing the writer remove association recall.
- Iterative versus direct maximum echo difference: approximately 1.33e-11.
- Direct normalized matrix equivalence error: 0.
- Largest upward energy step: floating-point roundoff, approximately 1.67e-16.

On the initial local run at load 32, Python direct connection reads took about
14.7 microseconds/query, Hebbian/delta about 11, and cache attention about 30.
Eighty-step relaxation took about 755 microseconds. These include Python input
validation and conversions and are illustrative, not reproducible hardware
rankings. The JSON includes local environment and measured timings. Scores,
not runtime metadata or SVG timestamps, should reproduce exactly with the
same NumPy behavior.

## Verdict and next question

**Mechanism gate passes. Distinctive algorithmic-advantage gate does not.**
The exact matrix equivalence was expected before running, not discovered as
a post-hoc excuse. A richer architecture might behave differently, but that
would require its own test. The current result supports queryable persistent
association memory. It does not establish unsupervised object segmentation,
semantic learning, compression of arbitrary images, brain mechanisms or an
advantage over transformers.

The next useful question is whether selective writes can preserve unrelated
bindings under a fixed memory budget. The delta baseline already sets the
minimum standard. Retain that baseline, use new seeds, and precommit the
changed/retained tradeoff before testing another writer.

## Reproduce

```bash
python -m unittest discover -s tests -v
node tests/test_demo.mjs
python -m aibrain.benchmark --output runs/gate1
```

[Complete checked-in receipts](gate1_metrics.json) · [Figure](gate1.svg) ·
[Original design](superpowers/specs/2026-10-19-queryable-memory-design.md)

[Verification record and remaining browser QA limitation](VERIFICATION.md)
