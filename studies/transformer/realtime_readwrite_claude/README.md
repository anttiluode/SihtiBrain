# RealTimeReadWriteClaude

**Where should a live memory look? Find the address a frozen language model already computes.**

[RealTimeReadWrite](https://github.com/anttiluode/RealTimeReadWrite) gave a frozen model a live delta-rule
memory (`S ← αS + β(v − Sk)kᵀ`, store only the surprise). It failed on **addressing**, not on plasticity:
the stored values were right (96% decodable), but a question's key matched the right fact no better than a
distractor (cosine 0.321 vs 0.318). The memory knew *what* but not *where*.

The Tolman–Eichenbaum Machine (Whittington et al., Cell 2020) says what an address must do: a structural
code `g`, path-integrated over relations, separate from sensory content `x`, such that **arriving at the same
place by a different route gives the same address**. Language models are known to bind entities to
attributes through abstract binding-ID vectors (Feng & Steinhardt, ICLR 2024). So before building `g`, this
repo asks whether the frozen model already has it.

Run the post-result diagnostics with `python gate0b_diagnostics.py --model Qwen/Qwen2.5-0.5B --out results/gate0b_qwen2.5-0.5b.json`.

The sibling line, AIbrainV2 Gate 2 (Sol), tests TEM-style addressing in small graph worlds on CPU. This
repo is the language half.

## Gate 0 — same entity, two routes ([PREREG.md](PREREG.md), committed before any code)

Context: 8 bindings such as `Anna's brother is Ben. Clara's boss is Dina. …` At every layer, read the
hidden state at the last token of `…Now consider Ben` (direct) and `…Now consider Anna's brother`
(relational). Does the relational state retrieve Ben among the 8 targets?

* Layer chosen on 50 dev worlds; scored on 150 held-out test worlds (1,200 queries).
* **PASS**: ≥ 80% centred top-1 accuracy at that layer (chance 12.5%) **and** ≤ 25% when the context
  only lists the names, with no relation sentences.
* Reported too: uncentred accuracy, and a 16-way test that asks whether "Anna's brother" sits closer to
  Ben or to Anna.

A pass gives a usable key **in context**. The memory gate proper (write, clear the KV cache, ask by the
other route) is Gate 1, to be pre-registered after Gate 0's result.

## Run

This environment cannot reach HuggingFace, so the gate has not been run yet. On a machine that can:

```bash
pip install torch transformers numpy
python -m unittest discover -s tests -v          # CPU, no download (miniature random Qwen2)
python gate0_route_probe.py --model Qwen/Qwen2.5-0.5B --out results/gate0_qwen2.5-0.5b.json
```

About 8,000 short forward passes; a few minutes on a GPU. Reported, not gated:
`--model Qwen/Qwen2.5-0.5B-Instruct` and `--model Qwen/Qwen2.5-1.5B`.

The script prints the chosen layer, test accuracy with a 95% Wilson interval, the control, the decision,
and a per-layer table. Everything goes into the JSON.

## Status

| Gate | Question | Status |
|---|---|---|
| 0 | Does a frozen LLM give "Anna's brother" and "Ben" the same address in context? | **FAIL** on Qwen2.5-0.5B: 41.4% at layer 14 (chance 12.5%, control 12.9%, need 80%). Real signal, not a usable key. [RESULT_GATE0.md](RESULT_GATE0.md) |
| 0b | Post-result: does the model resolve the two-hop at all, at which position, in which subspace? | Model answers Ben 77% (8-way 82%). A dev-fitted linear map recovers Ben from the "brother" state 87% at layer 17 (cosine there: 26%); controls 11–15%. The address exists in a subspace. [RESULT_GATE0B.md](RESULT_GATE0B.md) |
| S | Does a sparse nonlinear expansion fix Sol's Gate 2 address (82% → ?) | **PASS**: 100% new route, 99.6–100% retention; linear control stays at 82%. Equals the indexed table, does not beat it. [RESULT_SHARP.md](RESULT_SHARP.md) |
| 1 | Bindings and facts read once, context discarded, then "Where does Anna's brother live?" answered from two live memories keyed on the frozen model's own states | **FAIL**: 61.8% two-hop (chance 12.5%, controls 12.1% / 6.0%, need 80%). Hop 2 works (one-hop 98.3%); all loss is in hop 1's cue match. The model itself, context intact: 22.9%. Old sentence-end keys: chance. [RESULT_GATE1.md](RESULT_GATE1.md) |
| 1b | Hop 1 keyed on start + move (anchor ⊕ phrase), in worlds where every anchor and relation appears twice | **FAIL by 0.9 pts**: 79.1% [76.2, 81.8] two-hop (need 80%); outer-product key; phrase-only 32.6%, anchor-only 44.8%; model in context 22.8%. Dev–test (held-out names) gap 15 pts. [RESULT_GATE1B_2.md](RESULT_GATE1B_2.md) |
| 2 | The retrieved city state is added back into the residual stream; does the frozen model itself say the city, context empty? | **FAIL**: oracle write 48.5% 8-way but 0% top-1; memory write 34.0%. The write steers but the model never says the word: the stored state was 'just read Oslo', not 'about to say Oslo'. [RESULT_GATE1B_2.md](RESULT_GATE1B_2.md) |
| 2b | Store the model's own 'about to say it' state (rehearsal "Ben lives in Oslo. Ben lives in") as the value; does the model then say the city as its top-1 word? | pre-registered ([PREREG_GATE2B.md](PREREG_GATE2B.md)), code + tests in, **not run**. `python gate2b_speak.py --model Qwen/Qwen2.5-0.5B --out results/gate2b_qwen2.5-0.5b.json` |
