# SihtiStack — a causal return-path laboratory

**Can a fact influence a transformer's middle layer, disappear from its
answer, and be made visible again by returning a stored residue to that depth?**

This project is an *instrumented experiment*, not another associative memory
architecture. It connects [SihtiImageModel](https://github.com/anttiluode/SihtiImageModel)'s
saved-residue idea, [SihtiBrain](https://github.com/anttiluode/SihtiBrain)'s
branch/ping intuition and
[RealTimeReadWriteClaude](https://github.com/anttiluode/RealTimeReadWriteClaude)'s
frozen-model memory question. These other repositories are not modified.

## The simple core

An additive residual stream obeys:

```text
h_(l+1) = h_l + delta_l

h_last_before_final_norm = h_embedding + sum_l delta_l
```

That is exact bookkeeping. **It does not mean the final state is a lossless
record of all individual deltas.** Different sequences of deltas may have the
same final sum. To recover the history, you must record each delta, as Sihti
records bands. Final normalization and limited output readout add further
qualifications.

Our proposed **return** does something different from inverse reconstruction:

```text
donor  = middle_hidden("Fact: Anna lives in Oslo. ... Answer:")
base   = middle_hidden("Question: Where does Anna live? Answer:")
memory = donor - base

middle_hidden(fact_free_question) += 1.0 * memory
# now run the upper layers and inspect the next-token answer
```

The surrounding prompt and attention K/V states are **not** transferred.
The model must express the answer using a last-token vector alone.
The donor vectors use explicit subject labels in the test harness; this is
not native learned addressing. Frozen-model activation steering is established
prior art; SihtiStack measures whether this particular causal return works.

## Run

```bash
pip install -r requirements.txt
python -m sihtistack.ledger_demo
pytest -q

# Pre-registered default: Qwen2.5-0.5B-Instruct, layer 14, 12 names
python -m sihtistack.qwen_gate

# Optional SLOW oracle sensitivity measurement
python -m sihtistack.qwen_gate --gradient
```

Qwen model weights download on first real run; the real experiment has **not**
been run on pretrained weights in this PR. A single RTX 3060 should be sufficient
for this 0.5B test. Use `--device cpu` without CUDA (slower).
The runner assumes a Hugging Face decoder with `model.layers` and Qwen-like
decoder outputs; it does not support every transformer architecture.

Individual trials and aggregates are saved as `results/qwen_gate.json`.
The five arms are:

| Arm | Question | Intervention |
|---|---|---|
| none | Fact-free | No vector |
| in_context | Fact in prompt | No vector |
| return | Fact-free | Donor minus neutral state |
| wrong_city | Fact-free | Same name, **wrong** city donor |
| random | Fact-free | Norm-matched random vector |

Readouts: candidate-restricted city accuracy, actual full-vocabulary
next-token accuracy, and target-vs-alternative logit margin. Candidate
accuracy is **not** ordinary text generation accuracy.

The optional gradient diagnostic differentiates the *correct-city minus
wrong-city* logit margin with respect to the post-block state. It is
**answer-labelled**, therefore **oracle-only**, and not an inference-time
memory-learning strategy.

## Why this differs from Claude's real-time memory repo

Claude's work tests **key formation**, multi-hop composition, and live
external memory updates. This deliberately supplies the subject key so that
a failed address does not obscure a successful or failed **return channel**.
The goal is to isolate *where* information can be read or causally written.

If Gate 1 passes, next test held-out paraphrases, answer-position changes,
overwrites, and missing K/V context. If it fails, do not silently tune the
injection amplitude until answers flip: examine decodability, donor strength
and oracle sensitivity first.

## Evidence status

- **PASS:** algebra ledger reconstruction and demonstration that
  different histories can share one final state.
- **PASS:** nine local tests for telescoping, gradient decomposition,
  state hooks, city-token margin indexing, and a synthetic end-to-end
  five-arm return harness (no downloaded model weights).
- **UNRUN:** Qwen2.5 pretrained-model memory/return experiment.
- **UNPROVEN:** multi-hop, automatic native memory addressing, persistent
  synaptic learning or biological gradient descent.

See [PREREG.md](PREREG.md) for the fixed settings and interpretation rules.
