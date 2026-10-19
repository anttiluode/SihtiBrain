# PREREG — Gate 1: two-hop recall from a live memory after the context is gone

*Claude (Opus 5.5), 2026-10-19. Committed before any Gate 1 code and before any run. Builds on Gate 0
(FAIL, 41.4%), Gate 0b (post-result: the referent is linearly present in mid-to-late layers) and the
sharp-address result (a sparse expansion turns a blurry address into a usable key).*

## The task

A world is 8 bindings and 8 facts, read once as one chunk, sentence order shuffled:

```
Anna's brother is Ben.   Ben lives in Oslo.   Clara's boss is Dina.   Dina lives in Rome. ...
```

The chunk is processed by frozen **Qwen2.5-0.5B**, the memory is written, and **the context is then
discarded**. Each question is a fresh forward pass with nothing else in it:

```
Where does Anna's brother live?
```

The answer is a city the question never mentions, reachable only by two hops: Anna's brother → Ben → Oslo.
Score: 8-way choice among the world's 8 cities. Chance 12.5%.

## Held-out names

The 100-name list is split: names 0–59 are **dev**, names 60–99 are **test**. Dev worlds (seeds 0–49)
use only dev names; test worlds (seeds 1000–1099, 800 questions) use only test names. Cities come from
a fixed list of 40 one-word cities, 8 distinct per world. Relations as in Gate 0.

## The machine (nothing in the model is trained)

States are read at layer `L`, at the **last token of a referring phrase**:

| state | where |
|---|---|
| `rel_write` | end of `Anna's brother` in a binding sentence, in the chunk |
| `name_bind` | end of `Ben` in the same binding sentence |
| `name_fact` | end of `Ben` in a fact sentence |
| `rel_query` | end of `Anna's brother` in the context-free question |

Key map, fitted on dev worlds only:

```
K(h) = unit( top_k( ReLU( R · P( h − μ_type ) ) ) )
μ_type : dev mean of that state type      P : PCA to d components (optionally whitened)
R      : fixed Gaussian 2048 × d (seed 7)  k = 32
```

Two delta memories, each sentence written **once**, in reading order, β = 1:

```
M1 (bindings):  K(rel_write) → K(name_bind)
M2 (facts):     K(name_fact) → one-hot city
read:  v = M1 K(rel_query);   v' = unit(top_k(ReLU(v)));   city = argmax over the world's 8 cities of M2 v'
```

No string, name ID or token ID passes between the hops; the only link is the frozen model's own states.

## Selection, then one test

On dev worlds, μ and P are fitted on seeds 0–29, and the configuration is chosen on seeds 30–49 by
two-hop accuracy from the grid L ∈ {14, 17, 18} × d ∈ {64, 128, 256} × whitening ∈ {yes, no}. The chosen
configuration is refitted on seeds 0–49 and run **once** on the test worlds.

## Decision

**PASS** if, on the 800 test questions:

* two-hop accuracy **≥ 80%**, and
* the **no-memory** control ≤ 25%, and
* the **shuffled-binding** control ≤ 25%.

Otherwise **FAIL**.

| control | what it is |
|---|---|
| no memory | the model alone, question only: `…live? A: Anna's brother lives in` → 8-way over the world's cities |
| shuffled binding | M1 written with the 8 names permuted within the world |

## Reported, not gated

* **In-context ceiling**: the whole chunk plus the question in context, the model answers by itself.
  This is what the model does when nothing is forgotten.
* **One hop**: `Where does Ben live?` through M2 alone (key = end of `Ben` in the question).
* **Raw keys**: the same pipeline without the expansion (K = unit(P(h − μ))).
* **Sentence-end keys**: the naive RealTimeReadWrite-style memory, keyed on the state at each fact
  sentence's final "." and queried at the question's "?", one hop to the city.
* **Oracle table**: exact strings, `"Anna's brother" → "Ben" → "Oslo"`. 100% by construction; it shows
  what a symbolic store gets for free.
* Per-hop accuracies: does hop 1 land on the right name key (nearest of the world's 8 `name_fact` keys)?

## What a pass would and would not show

A pass shows that a frozen model's own states, read at the right positions and sharpened, are good
enough keys for a live external memory to answer a two-hop question after the context is gone, with no
fine-tuning, no retrieval over text and no symbolic IDs. It would not show the memory beats a
retrieval system that stores the text itself; that comparison comes later, if this passes.
It would not show anything about long streams, many worlds in one memory, or other templates.
