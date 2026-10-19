# PREREG — Gate 1b (start + move address) and Gate 2 (the model speaks from memory)

*Claude (Opus 5.5), 2026-10-19. Committed before any code for these gates and before any run.*

Gate 1 ([RESULT_GATE1.md](RESULT_GATE1.md)) failed at 61.8% two-hop, with all of the loss in hop 1: the
single state at the end of "Anna's brother" is a poor cue across contexts. Hop 2 worked at 98.3%.

## A flaw in Gate 1's worlds, fixed here

In Gate 1 every anchor appeared in exactly one binding, so the anchor name alone identified the
binding and the relation word carried no information. A real address test needs both parts to matter.

**New worlds:** 4 anchors × 2 relations = 8 bindings. 4 relations are drawn per world; anchor *a*
gets relations *a* and *a+1* (mod 4), so **every anchor appears twice and every relation appears
twice**:

```
Anna's brother is Ben.   Anna's boss is Dina.   Clara's boss is Emma.   Clara's cousin is Finn. ...
Ben lives in Oslo.       Dina lives in Rome.    ...
```

An anchor-only key or a relation-only key can resolve at most half of hop 1. Only the conjunction
(start point + move, TEM's form of an address) resolves all of it. 8 targets, 8 cities, 16 sentences
read once in shuffled order. Names split as in Gate 1 (dev names 0–59, test names 60–99); test worlds
are seeds 2000–2099 (800 questions); dev fit seeds 0–29, dev select seeds 30–49.

## Gate 1b — hop 1 keyed on start + move

States from frozen Qwen2.5-0.5B, read at the last token of: the anchor name (`Anna`), the referring phrase
(`Anna's brother`), the target name in the binding and in the fact, in the chunk; and the anchor and the
phrase in the context-free question.

Each part gets its own dev-fitted map: per-type mean, PCA to *d*, whitened. The hop-1 key combines
the anchor part and the phrase part:

* **concat:** `[z_anchor ; z_phrase]`
* **outer:** `vec(z_anchor[:d/4] ⊗ z_phrase[:d/4])` (TEM's conjunction)

followed by the sparse expansion (2048 units, top-32) or plain unit normalisation. Memories and reads
are exactly as in Gate 1 (delta, β = 1, one pass; hop 1 → clean-up → hop 2).

Selection on dev select worlds by two-hop accuracy over compose ∈ {concat, outer} × layer ∈ {14, 17} ×
d ∈ {32, 64} × expansion ∈ {yes, no}. One test run.

**PASS** if two-hop ≥ **80%**, no-memory control ≤ 25%, shuffled-binding control ≤ 25%.

Reported: the Gate 1 phrase-only key and an anchor-only key on the new worlds (both expected near or
below 50% at hop 1), hop-1 accuracy, one-hop, the in-context ceiling.

## Gate 2 — write the memory back into the stream and let the model answer

A third delta memory stores, under the same name keys as hop 2, the centred layer-*Lv* state of the
city token in each fact sentence (`Oslo` in "Ben lives in Oslo."). At question time, after hop 1:

```
m = M3 · clean(M1 · key_query)                       retrieved city state
h_Lv[last] ← h_Lv[last] + α · ‖h_Lv[last]‖ · unit(m)  added to the residual stream at the last token of
                                                     "Q: Where does Anna's brother live?\nA: Anna's brother lives in"
```

The remaining layers run unchanged and the model's own next token is scored: 8-way over the world's
cities (first token), and full-vocabulary top-1. Context is empty; the only route to the city is the write.

Selection on dev select worlds over Lv ∈ {6, 10, 14} × α ∈ {0.5, 1, 2, 4} by memory-injection 8-way.

**PASS** if all four hold on the test worlds:

| | criterion |
|---|---|
| (a) oracle write: inject the true city's stored state, skipping the memory | ≥ 80% |
| (b) memory write | ≥ 0.85 × Gate 1b's two-hop accuracy |
| (c) wrong-city write: inject another city of the world; rate of answering the *right* city | ≤ 25% |
| (d) no write | ≤ 25% |

(a) asks whether writing down works at all; (b) whether speaking loses little beyond what the memory
already lost; (c) whether the model follows the write rather than guessing.

Reported: full-vocabulary top-1 for every condition, and the in-context ceiling.

## What a double pass would mean

A frozen 0.5B model reads a text once, the text is discarded, and asked a question that needs two facts
it never saw together in one question, it says the right city itself. The model's own states are the
addresses, a small external memory composes them, and the answer goes back into the stream at the depth
it came from. It would not show scale, long streams, many worlds sharing one memory, or other templates.
