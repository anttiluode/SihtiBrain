# Gate 1 result — does equilibrium geometry make memory more retrievable?

2026-10-19. The [protocol](GATE1_PROTOCOL.md) was committed before implementing the experiment. The degree-normalized diagnostic and paired uncertainty analysis were added **after the initial benchmark result**; neither was in the preregistered stopping gate.

## Main result

Same *16×8 float32 matrix* (512 bytes). Same add-only writer, exactly equal conductance/weight matrices after **every** matched write and rewrite. Different readout only:

- Direct: `scores = cue @ W`
- Graph: inject cue into 16 input nodes of a 24-node grounded passive graph and read its eight value-node equilibrium voltages.
- Indexed control: packed 16-bit cue + 8-bit value; update existing key directly; nearest-Hamming read. Its footprint is only 3 bytes per binding here and its strong performance is an important practical baseline, **not** evidence for the graph.

Twelve independent random seeds, with 8/16/32/64 associations, 3 presentations per write, 25% of stored bindings reassigned and written three more times, with one-active-bit-swap noisy cues held out from writing.

| Associations | Graph retention after rewrite | Direct matrix | Post-result degree-normalized matrix | Indexed control |
|---|---:|---:|---:|---:|
| 8 | 81.5% | 69.0% | 82.4% | 90.7% |
| 16 | 62.5% | 48.4% | 56.7% | 81.7% |
| 32 | 44.3% | 37.2% | 42.5% | 70.9% |
| 64 | 34.0% | 27.1% | 31.9% | 52.3% |
| **Mean** | **55.6%** | **45.4%** | **53.4%** | **73.9%** |

The average graph advantage against direct weights is **+10.17 percentage points** on retention. For newly rewritten labels, graph was **39.97%**, direct **40.67%**, averaged over loads and seeds. This satisfies the committed threshold (+5pp retention and no more than −2pp new-label regression). But the absolute performance is poor at high loads, and indexed memory wins despite using **24, 48, 96, or 192 bytes**, rather than 512.

### Important post-result explanation

Let `G` denote the nonnegative 16×8 matrix, and `D_k=diag(leak+sum_value G)`, `D_v=diag(leak+sum_cue G)`. The grounded Laplacian is

```
H(G) = [D_k  -G  ]
       [-Gᵀ  D_v ]
```

The exact query-to-value Green's function is

```
T(G) = [D_v - Gᵀ D_k⁻¹ G]⁻¹ Gᵀ D_k⁻¹.
```

The graph settles to `values = T(G) @ cue`. This is simply an effective matrix transformation, not a new information channel. We validated the Schur-complement result against a complete 24×24 linear solve and verified that Euler relaxation converges to the same values.

Crucially, the **first-order no-relaxation approximation**

```
T₀(G) = D_v⁻¹ Gᵀ D_k⁻¹
```

already reaches **53.4%** mean noisy retention with exactly the same stored matrix and no equilibrium solving. That captures the great majority of the observed improvement over naive `cue @ G`. The full nonlinear Schur correction adds only about **2.2 percentage points** on this metric. Both normalizations and passive propagation are conventional methods, and this gate does not establish a new class of memory.

A paired per-seed bootstrap with 10,000 fixed-seed resamples is printed by `demo.py`; it was added after seeing the initial data and is descriptive, not preregistered.

## Structural controls

- Fixed nonnegative conductances and positive leak make the operator symmetric positive definite.
- The two paired read architectures store byte-for-byte identical matrices and use the same writer.
- Clearing fast activity leaves readouts unchanged; erasing connections eliminates graph readout signal.
- Cue response is linear in the cue for a fixed graph (`T(G)`); changing conductances changes `T` nonlinearly.
- The indexed baseline overwrites an old label for the same key and retains unrelated labels by construction; unlike a distributed matrix, it supports explicit addressing.
- The cue-label mapping is deliberately random. There is no held-out *unseen label assignment* or automatic semantic generalization. Held-out queries are **corrupted versions of stored cues**.

## Interpretation and stopping point

**Passed:** passive response geometry improves noisy retention relative to a naive direct read of the exact same stored matrix, under this particular fixed Hebbian writer and benchmark.

**Not demonstrated:** an improvement over optimized/normalized direct readers, superior density or scaling, learning semantics, better image segmentation, or a neurobiological mechanism. In fact, a compact indexed memory beats every fixed 512-byte matrix in this random-key task.

This gate suggests an important next question: is the geometry's advantage robust under a **strict post-result normalization baseline**, with selective correction rather than add-only writes? That must be designed and preregistered as a fresh gate, not post-hoc adjustment of Gate 1.

Full receipts and figure: `python demo.py`; GitHub Actions uploads both from Python 3.11 and 3.13.
