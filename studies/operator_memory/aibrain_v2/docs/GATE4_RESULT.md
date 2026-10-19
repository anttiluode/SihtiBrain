# Gate 4 result — sparse expansion removes most crosstalk; sensory correction is conventional filtering

**2026-10-19.** The [Gate 4 preregistration](GATE4_PROTOCOL.md) was committed before running the full 20-world experiment. The implementation follows [Gate 3](GATE3_RESULT.md) on the same eight source-world learned automaton and twenty held-out worlds. It uses a **12-dimensional Laplacian spectral code from Gate 3**, *not literally the six hand-selected Fourier modes from Gate 2*; both describe a small, already injective state code in a shared relational world.

## What Claude predicted

A low-dimensional structural address can identify the location exactly but, when used directly as an associative key, overlaps with other addresses. Frozen nonlinear place-cell-style expansion may make the *readout* easier to separate:

    p(g) = L2normalize(topK(ReLU(R g)))

where R is a fixed seeded Gaussian projection, k=8 for 64 units or 16 for 256 units. Training and inference both use the **same** frozen R and the graph-address code learned in Gate 3. Only a delta-rule value memory is written; the feature map is **not supervised or tuned after seeing these results**. We store independent arbitrary label values for visited locations, and revisit by a *different route*.

It is not strictly correct to say that any missing information is reconstructed. The 12D structural code distinguished all 36 states already; a deterministic nonlinear lifting cannot invert genuinely colliding addresses. Instead it creates a higher-dimensional, less-interfering representation. The exact form of the learned spectral codes is different from the hand-coded six-mode Fourier representation discussed by Claude.

## Primary preregistered gate: PASS

20 independent held-out worlds, 18 or 30 visited of 36, three writes, 25% of labels changed and written three more times:

| Address/readout | New-route recall at 18 | New-route recall at 30 | Changed label at 30 | Untouched retention at 30 | Total stored bytes |
|---|---:|---:|---:|---:|---:|
| Original 12D spectral / delta | 83.1% | 54.5% | 100% | 28.9% | 2,256 |
| Sparse 64D / delta (top-8) | 100% | 97.5% | 100% | 86.1% | 6,992 |
| **Sparse 256D / delta (top-16)** | **100%** | **100%** | **100%** | **98.7%** | **22,352** |
| Dense ReLU 256D / delta | 94.7% | 80.7% | 100% | 53.5% | 22,352 |
| Independent random-state 256D sparse hash | 100% | 100% | 100% | 100% | 45,200 |
| Exact one-hot 36D state / delta | 100% | 100% | 100% | 100% | 1,296 |
| **Learned graph + indexed value** | **100%** | **100%** | **100%** | **100%** | **174 at 30** |

The 256D top-k readout has an enormous, reproducible improvement over the original 12D readout and its equal-width dense activation control. **Primary sparse-key gate passes** under the predeclared criterion: new-route recall and untouched retention each improve by >10pp at 30 visited nodes, changed recall remains >95%, and the sparse 256D readout beats dense 256D.

**But this is not a victory over a proper address table.** A learned-graph one-hot readout needs only 1,296 bytes and is perfect. Explicit indexed state→value storage needs only the 144-byte shared action table plus 30 one-byte labels = **174 bytes**, and is also perfect. The independent sparse state-hash control is perfect too, but requires 45,200 bytes. Thus much of the useful ingredient is **near-orthogonality / sharpened addressing**; it isn't distinctive evidence for cortical place-cell biology or frequency-product physics.

Storage counts include *all* frozen spectral state codes (36×12 float32 = 1,728B), the random projection (256×12 float32 = 12,288B), the learned graph (36×4 uint8 = 144B), and the value readout (8×256 float32 = 8,192B). Random generation from a seed could compress some of this in a different implementation, but the same accounting convention is used for the checked-in model.

## Secondary preregistered gate: noisy odometry and sensory correction — PASS, not uniquely phase-based

Queries follow a genuine alternative route, while the *logged* action is replaced with a random different action with probability 15% or 30% per step. The true environment emits one of four **aliased** sensory symbols at every step (5% symbol corruption). Only the symbols previously observed at the training-visited nodes are stored as a sensor map. The reader never sees true coordinates or target labels at query time.

We compare a phase-key sparse memory against a **classical indexed graph with exactly the same Bayesian location filter**, including the full posterior value-decoding strategy. Each uses the **same frozen 36-state learned action automaton and same observed symbols**.

### Accuracy at 15% wrong action reports

| Reader | 18 visited | 30 visited |
|---|---:|---:|
| Sparse 256D, wrong-action open loop | 37.2% | 38.5% |
| Sparse 256D, Bayes MAP correction | **70.8%** | **83.3%** |
| Indexed lookup, wrong-action open loop | 31.7% | 36.8% |
| Indexed lookup, Bayes MAP correction | 68.3% | 82.7% |
| **Indexed graph, full value posterior** | **81.4%** | **87.3%** |
| Indexed graph, oracle true location (not deployable) | 100% | 100% |

### Accuracy at 30% wrong action reports

| Reader | 18 visited | 30 visited |
|---|---:|---:|
| Sparse 256D, open loop | 21.7% | 22.2% |
| Sparse 256D, Bayes MAP correction | 61.7% | 68.5% |
| Indexed lookup, Bayes MAP correction | 60.0% | 67.5% |
| **Indexed graph, full value posterior** | **75.0%** | **73.8%** |

At 15% action corruption, the sparse MAP-corrected memory improves over its same-model uncorrected condition by a mean **+39.2 percentage points** across loads. The *strongest available filter* is **the ordinary indexed value posterior**, which uses all the same evidence instead of committing early to a single MAP state.

**Interpretation:** noisy odometry can be corrected by sensory evidence even when individual landmarks are aliased. However, the win comes from **filtering and fusion**, not proof that phase-grid coordinates intrinsically correct noise better than symbolic coordinates. The full index posterior is superior, as it should be on arbitrary value labels and a small known graph.

With perfect action logs both methods reach 100%; the sensor noise has no effect on the exact state estimate then because the transition model remains deterministic.

## What this demonstrates and what it does not

**Demonstrated in this synthetic setting:**
- Frozen sparse nonlinear lifting of already-injective 12D structural keys reduces memory interference dramatically.
- Most of that benefit is key discrimination / high-dimensional separation; one-hot and indexed controls equal or beat it.
- A history-based filter fusing unreliable action reports and aliased observations can recover from localization error.
- Sparse and index memories can use the same action filter. The index plus full posterior remains the strongest measured method.

**Not demonstrated:**
- Neural or biological grid-to-place computation, or learned TEM.
- True restoration of information discarded by a noninjective sieve.
- State discovery from ambiguous source landmarks; source landmarks remain unique in this Gate 4.
- Equal-byte or equal-compute superiority over sparse hash, one-hot or indexed methods. The 256D sparse representation trades >100× indexed storage for similar clean recall.
- General advantages in real language-model memory, natural image segmentation or hippocampal dynamics.

**Next discriminating gate:** learn dynamics *from aliased/noisy source observations* or allow continuously changing worlds where symbolic indexing becomes expensive. Compare to particle/Bayesian filters and properly trained recurrent state estimators with equal access to actions, observations and memory. Do **not** merely inflate the lifted representation further to chase a percentage already at 100%.

## Reproduce

    python -m unittest discover -s tests -v
    python demo_place.py --output runs/place_gate4

The script writes a complete JSON record for every seed/load, \`summary.csv\`, and four-panel figure \`place_gate4.png\`. GitHub Actions runs all gates (1–4) and reports **57 passing tests** on Python 3.11 and 3.13, publishing reproducibility artifacts.
