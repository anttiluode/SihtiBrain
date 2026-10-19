# AIbrainV2 — Queryable response geometry

![Queryable operator memory](docs/response_geometry.svg)

**Can experience change how a system answers a query, without continuously maintaining an activation?** This small, CPU-only associative-memory laboratory makes that idea testable.

Input a 16-bit cue (four active bits). Store one of eight labels in a **16×8 conductance matrix**. A graph reader injects the cue into a passive network and asks which of eight value nodes responds most strongly. A direct fast-weight reader uses **the exact same 512 bytes** and **the exact same Hebbian write operations**, but reads with `cue @ W`. A packed indexed lookup is a strong alternative, with its actual storage footprint shown.

This is deliberately **not** an artificial brain simulator or an unsupervised object segmenter. The architecture follows the most solid part of the SihtiBrain lineage: local changes can persist in a response operator and be revealed by a later query.

## Run it

Requires Python 3.10+:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python demo.py --output runs/operator_gate1
python gui.py
```

On Windows, double-click `run_gui.bat` after installing dependencies.

In the GUI, choose **exactly four** blue cue squares and a value 0–7, then **Write association**. Try **Query**, **Swap cue bit**, and writing a *different* value to the same cue. **Reset activity** leaves the connection memory intact; **Erase all connections** removes it. The graph and direct weights always receive identical writes. The indexed baseline deliberately overwrites that key.

## Gate 1: preregistered experiment

[Precommit protocol](docs/GATE1_PROTOCOL.md) · [Full results, controls and interpretation](docs/GATE1_RESULT.md)

Twelve independent seeds; 8, 16, 32, 64 random associations; held-out one-bit-swap noisy queries; then rewrite 25% of the values and test both changed and unchanged bindings.

### Noisy recall of untouched associations after rewrites

| Associations | Graph response | Direct same-matrix read | Degree-normalized (post-result) | Indexed lookup |
|---|---:|---:|---:|---:|
| 8 | 81.5% | 69.0% | 82.4% | 90.7% |
| 16 | 62.5% | 48.4% | 56.7% | 81.7% |
| 32 | 44.3% | 37.2% | 42.5% | 70.9% |
| 64 | 34.0% | 27.1% | 31.9% | 52.3% |
| **Average** | **55.6%** | **45.4%** | **53.4%** | **73.9%** |

The preregistered main gate **passes** (+10.17 percentage points retention against direct fast weights, while rewritten-association recall regressed only ~0.7 points). But the **post-result** degree-normalized matrix nearly matches the full graph. Thus most of the benefit comes from well-known degree normalization; this is not evidence of a new form of AI memory. And for random discrete keys, a compact indexed table is substantially better, storing just **3 bytes per binding** against either matrix's fixed 512 bytes.

### What exactly is the graph calculating?

Given nonnegative conductances `W`, all 24 nodes have leak `λ=0.25`. With `D_k=diag(λ+row_sum(W))` and `D_v=diag(λ+col_sum(W))`,

```
H(W) = [ D_k  -W  ]
       [ -Wᵀ  D_v ]
```

The cable minimizes `E(z)=0.5 zᵀH(W)z - inputᵀz`. Its settled read is

```
value_response = T(W) @ cue
T(W) = (D_v - WᵀD_k⁻¹W)⁻¹ WᵀD_k⁻¹.
```

This is an ordinary grounded-Laplacian Green's function (Schur complement). Iterative gradient relaxation and an exact matrix solve return the same result. At fixed W the read is linear in the query. Memory consists of changing **W**, so future reads change even after voltages are cleared.

The simple, no-relaxation first-order approximation `D_v⁻¹WᵀD_k⁻¹` was added as an explanatory baseline **after** observing the initial outcome; it captures most of the improvement.

## Gate 2: TEM-inspired relational addresses (new independent experiment)

**[Precommitted protocol](docs/GATE2_PROTOCOL.md) · [Detailed results and falsification](docs/GATE2_RESULT.md)**

Does an action-integrated address retrieve a previously experienced location through a route never taken before? Unlike Gate 1, the tasks have reusable relational structure. A given 6×6 torus has arbitrary independent sensory symbols and labels; the agent sees action sequences and observed values, not the target coordinate ID.

Across twenty fresh worlds and 18 observed locations, a **384-byte phase/delta memory** recalls **82.2%** of hidden labels via different routes, versus **34.2%** for content keys and **18.3%** for time keys. All share the same memory size. The action-rotation invariance passes to numerical precision; a wrong action rule destroys the advantage. Thus the **structural-addressing gate passes**.

**But the decisive strong control wins:** a classical indexed lookup integrates the *same given action structure*, recalls **100%**, and uses only **54 bytes** at this load. At 30 visited locations, phase recall falls to **56.7%**, versus indexed **100%** in 90 bytes. The *algorithmic-superiority gate fails*. We supplied the transition algebra instead of learning it, so this is NOT an implementation of the learned Tolman–Eichenbaum Machine.

The rewrite results expose a useful memory tradeoff: at 18 locations, phase-delta correctly writes **100%** of changed labels but preserves only **66.1%** of untouched ones; phase-Hebb preserves **84.3%** but updates only **46.3%** of changed labels. The indexed baseline achieves both at 100%.

```bash
python -m unittest discover -s tests -v
python demo_tem.py --output runs/tem_gate2
```

The four-panel figure, complete twenty-world JSON and summary CSV are reproducible locally and uploaded by GitHub Actions. **Next**: separately test whether an action structure can be *learned* and reused across unseen worlds, against proper structured graph/index and learned-coordinate baselines, before connecting it to a frozen LLM.

## Gate 3: learn the transition graph from experience, then transfer

**[Precommitted protocol](docs/GATE3_PROTOCOL.md) · [Results, limitations and post-result seed check](docs/GATE3_RESULT.md)**

Gate 2 was handed a torus transition rule. Gate 3 instead **identifies a 36-state action automaton from action-labeled walks past unique, randomly renamed source-world landmarks** and computes structural keys from its graph spectrum. All eight source worlds recover the same canonical state-transition table; the action structure then transfers to **20 held-out worlds** with four aliased sensory symbols and fresh arbitrary target labels.

New-route recall at 18 visited locations: learned graph + spectral/delta **83.1%**, content-only **34.2%**, time-only **18.3%**. This passes the structural-addressing gate. Yet the **same learned graph with an indexed table** recalls **100%** and requires **162 bytes**, compared with **2256 bytes** total for spectral fast weights (including graph, code table, and weights). It decisively wins on rewrites and untouched retention.

The spectral read only narrowly wins over one random graph-state code at 18 visits (83.1% vs 77.8%). A separately labeled **post-result** check across 16 random code seeds shows an average **+3.3-point** gain at 18 visits that becomes a **−1.6-point** disadvantage at 30 visits. Spectral geometry is not established as an algorithmic breakthrough.

This is **classical system identification with uniquely identifiable landmarks**, not a learned TEM neural network: the transition model is constructed symbolically, the target worlds share action semantics, and the reset anchor is provided. The next research question is discovering reusable structure from **aliased partial noisy observations** without relying on landmark IDs.

\`\`\`bash
python demo_learned.py --output runs/learned_gate3
python diagnostic_gate3_random_seeds.py --output runs/learned_gate3
\`\`\`

## Gate 4: sparse structural keys and recovery from noisy actions

**[Precommitted protocol](docs/GATE4_PROTOCOL.md) · [Full results and limitations](docs/GATE4_RESULT.md)**

Claude's suggestion prompted a tightly controlled place-cell-style experiment: transform the existing 12D learned spectral graph address with a *frozen* random projection followed by **ReLU + top-k sparsity**, then write value labels through the same delta fast-weight rule as Gate 3. Independently test noisy action logs, sensory correction from partially known four-way aliased landmarks, and **a structured indexed graph using the same Bayesian filter**.

Across 20 held-out worlds with 30 observed nodes and 25% rewritten labels:

| Memory | New-route recall | Untouched retention | Total state bytes |
|---|---:|---:|---:|
| Original learned spectral 12D | 54.5% | 28.9% | 2,256 |
| Sparse 64D (top 8) | 97.5% | 86.1% | 6,992 |
| **Sparse 256D (top 16)** | **100%** | **98.7%** | **22,352** |
| Dense ReLU 256D | 80.7% | 53.5% | 22,352 |
| Fixed random 256D state hash | 100% | 100% | 45,200 |
| One-hot state (36D) | 100% | 100% | 1,296 |
| **Indexed learned graph** | **100%** | **100%** | **174** |

The **precommitted sparse-key separation gate passes**, but this is a memory-capacity tradeoff, *not an efficient win over ordinary indexed addressing*. All 36 source locations were already distinct in 12D: nonlinear expansion improves separability without recovering truly lost information.

With **15% corrupted action reports**, sparse 256D recall rises from **37.2% to 70.8%** at 18 stored locations when corrected by a Bayesian sensor filter. **An indexed graph supplied with the same sensor observations and a full value posterior achieves 81.4%**. At 30 stored locations the corresponding figures are 38.5%, 83.3%, and 87.3%. Sensor fusion, not an exclusive property of phase codes, explains the improvement.

The learned source graph still depends on **uniquely identifiable source landmarks**; Gate 4 does not learn transitions from aliased input or implement the full Tolman–Eichenbaum Machine.

\`\`\`bash
python -m unittest discover -s tests -v
python demo_place.py --output runs/place_gate4
\`\`\`

The source, 57 invariant tests across Gates 1–4, JSON results, CSV, and figure are reproduced in GitHub Actions on Python 3.11 and 3.13. Next: learning a world state under **aliased, noisy source observations**, with a classical Bayesian filter and learned recurrent comparators.

## What's next?

Not another simulated neuron. A meaningful next gate would keep the same budget but test **selective error-correcting writes** against a strong normalized fast-weight control and indexed memory, on longer sequences with harder interference. Freeze the protocol first; let the result be negative if it fails.

Related experimental lines: [SihtiBrain](https://github.com/anttiluode/SihtiBrain), [SihtiBrainV2](https://github.com/anttiluode/SihtiBrainV2), [SihtiBrainV3](https://github.com/anttiluode/SihtiBrainV3), [GraphNN](https://github.com/anttiluode/GraphNN), [AIBrain](https://github.com/anttiluode/AIBrain).

All code is small, standard numpy + optional matplotlib/Tkinter. The benchmark outputs a four-panel PNG and full JSON (all seeds, loads, rewritten scores, exact/noisy macro recall). GitHub Actions tests Python 3.11 and 3.13 and archives the data.
