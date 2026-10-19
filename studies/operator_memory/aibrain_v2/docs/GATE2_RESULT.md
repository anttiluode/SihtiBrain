# Gate 2 results — relational addresses repair aliasing, but indexed memory still wins

**2026-10-19.** The [protocol](GATE2_PROTOCOL.md) was committed before the full 20-world run. This is a deliberately minimal **TEM-inspired addressing ablation**, *not* the Tolman–Eichenbaum Machine, learned grid cells, a frozen-LLM experiment, or a replication of brain data.

## Exactly what was constructed

- Each new world is a 6×6 torus whose 36 locations have random independent *target values* (0–7) and aliased sensory symbols (0–3). The same action algebra (N/S/E/W) is supplied to every world, but labels and symbols are resampled; there is **no learned structural model**.
- Phase keys encode six sin/cos Fourier modes (12 float channels) updated by multiplying action-conditioned rotations, rather than reading a node ID or integer coordinate. The matrix stores eight target-value outputs × 12 phase inputs as float32: **384 bytes**.
- Training sees only **18 or 30 locations** per world, repeated three times. Each query returns to an already observed location by a different action sequence and extra inverse-action detours. The target is *not* shown at query time. Unvisited locations are evaluated separately; their arbitrary target values cannot be deduced.
- All unstructured comparators use 384-byte matrices, identical update strength and observations; conventional indexed memory gets the *same known torus rule*, with an explicitly reported much smaller byte footprint. The edge-only graph cannot use edges not observed during training, whereas the indexed-coordinate baseline gets the entire known transition structure, just like the phase model.
- Twenty independent worlds, seeds 1000–1019, no validation selection. Tests and reproducible figure in GitHub CI.

## Preregistered gates

- **Gate 0 PASS:** the same location receives the same phase key along both paths (max error **1.67×10⁻¹⁶**), a closed action loop returns to the anchor (error **1.11×10⁻¹⁶**), and all 36 coordinate keys are distinguishable (minimum Euclidean separation **0.8165**). These are properties of the **given rotations**, not features the model discovered by experience.
- **Gate 1 PASS:** new-route recall at 18 visited locations is **82.2%** using phase + delta, versus **34.2%** content keys and **18.3%** time keys. With 30 visited locations, phase + delta still scores **56.7%**, although interference grows.
- **Algorithmic advantage over the strong indexed baseline: FAIL:** a classical action-integrated coordinate table retrieves **100%** of changed and unchanged visited labels, even under unseen query routes, storing **54 bytes** at 18 locations and **90 bytes** at 30, versus **384 bytes** for our phase matrix. The winner is addressing structure, not the matrix.
- **Unvisited value inference: unsupported as expected:** outputs at truly unobserved nodes hover near the 12.5% random-label floor. The indexed table returns a deliberate unknown rather than inventing a value; its reported accuracy is 0% under strict forced exact-label scoring, which should **not** be interpreted as worse calibrated uncertainty.

### Twenty-world summary

| Memory / address | New route / 18 visited | New route / 30 visited | Rewrite new value / 18 | Retain untouched / 18 |
|---|---:|---:|---:|---:|
| **Phase + delta** | **82.2%** | 56.7% | **100.0%** | 66.1% |
| Phase + additive Hebb | 85.0% | 70.2% | 46.3% | 84.3% |
| Content + delta | 34.2% | 21.0% | 72.5% | 19.6% |
| Time + delta | 18.3% | 16.0% | 7.5% | 15.7% |
| Wrong time-dependent action law | 17.2% | 16.3% | 10.0% | 20.0% |
| Indexed coordinate (same structural prior) | **100%** | **100%** | **100%** | **100%** |
| Observed edges only | 0% | 0% | 0% | 0% |
| No memory | 10.8% | 11.0% | 13.8% | 12.1% |
| Shuffled write labels | 15.3% | 15.3% | 6.2% | 16.1% |

At **30** visited locations, phase + delta has **98.6%** changed-binding recall but **32.2%** untouched retention; phase + Hebb has **42.9%** changed-binding recall and **68.5%** untouched retention. This is the old stability–plasticity dilemma in a particularly clear controlled setting.

### What the strong classical baselines mean

**A plain indexed table supplied with exactly the same action-to-coordinate structural prior solves the task perfectly while using far fewer bytes.** It is not an unfair oracle: its address is integrated from the action sequence, as the phase model's address is. The only difference is the representation/read-write mechanism. This convincingly rules out an advantage for the candidate on *arbitrary, independent* target labels in a small known torus.

The **edge-only** graph does not generalize, because the query sequences intentionally begin with unobserved detours. Zero here demonstrates the expected limitation of memorizing visited edges without a transition rule. Do not promote this deliberately restricted control above the prior-informed indexed table.

### Relation to TEM and Sihti

TEM (Whittington et al., 2020, *Cell*, doi:10.1016/j.cell.2020.10.024) learns a reusable structural representation across worlds and binds each new world's content into fast associative memory. **This gate provides the structure algebra by hand**, so it probes *addressability after structure is known*. It does not test how to learn reusable relational structure from observations, and it does not reproduce TEM's model or physiological predictions.

Our phase memory persists across cleared activation states; its state determines how later questions are answered, but no active dendritic propagation is simulated. The changed-versus-untouched tradeoff motivates **selective writes**, not more cortical analogies.

## Interpretation and next stop

We have now tested why pure content-derived/time-derived keys can fail on aliased structured tasks. A key based on the world transition structure is regenerable along routes not previously used. **This much is computationally valuable but already well known in structured memory and graph algorithms.**

The consequential **next gate** is to **learn** action/relation transition operators across worlds, rather than supplying them, and show held-out compositional recall. Fair baselines must include a learned symbolic graph model, BFS/indexed retrieval, ordinary delta memory with identical learned keys, and token/sequence-position encoding. After that, and only after it survives, consider a frozen LLM demonstration in RealTimeReadWrite.

### Run

```bash
python -m unittest discover -s tests -v
python demo_tem.py --output runs/tem_gate2
```

The run produces `metrics.json` (all forty cases), `summary.csv`, and `tem_gate2.png`. CI uploads all three. No GPUs or downloads required.
