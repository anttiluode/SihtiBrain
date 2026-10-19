# PREREG — Gate 0: does a frozen LLM reach the same address by two routes?

*Claude (Opus 5.5), 2026-10-19. Committed before any probe code exists and before any model is run.*

## Why this gate

[RealTimeReadWrite](https://github.com/anttiluode/RealTimeReadWrite) bolted a live delta-rule memory onto a
frozen model and failed on **addressing**, not on plasticity: values were stored correctly (96% decodable)
but the question's key matched the right fact no better than a distractor (cosine 0.321 vs 0.318).

The Tolman–Eichenbaum Machine (Whittington et al., Cell 2020) separates a structural address `g`
(path-integrated over actions/relations) from sensory content `x`, and binds them in a fast memory.
Its defining property: **arriving at the same node by a different route yields the same address.**

Feng & Steinhardt (ICLR 2024) found that language models bind attributes to entities through abstract
"binding ID" vectors. If a frozen LLM already computes something route-invariant, the live memory should
key on it instead of on content tokens. Gate 0 asks whether such a state exists, at which layer, and how
cleanly. It is inference only; nothing is trained.

## Stimuli

A **world** is 8 bindings. Each binding is one sentence `"{A}'s {r} is {B}."` with anchor name `A`,
relation `r`, target name `B`. All 16 names in a world are distinct, drawn from a fixed list of single-word
first names; relations from {brother, sister, friend, boss, neighbour, teacher, cousin, partner}.
Sentence order is shuffled. Context = the 8 sentences joined by spaces.

Every probe prompt is `context + "\nNow consider " + REF` and the hidden state is read at the **last token**
of the prompt (the last token of `REF`), at every layer (embedding output = layer 0).

| route | REF | count per world |
|---|---|---|
| direct-target | `B_i` | 8 |
| direct-anchor | `A_i` | 8 |
| relational | `A_i's r_i` | 8 |
| relational, no-binding control | `A_i's r_i`, but the context lists the 16 names with no relation sentences | 8 |

Worlds are generated from fixed seeds: **dev = seeds 0–49, test = seeds 1000–1149.**

## Metric

For a world and layer `l`: centre the 8 direct-target states by their world mean (the same mean is
subtracted from the relational states). Relational query `i` is **correct** if its nearest direct-target
state by cosine is `B_i`. Chance = 1/8 = 12.5%.

Reported alongside (not gating): uncentred accuracy; a 16-way version whose candidates are the 8 targets
plus the 8 anchors, with the fraction of relational queries landing on their own anchor `A_i` (is the
state "about" Anna or about Ben?); the margin cos(correct) − max cos(wrong).

## Decision rule

1. Choose the single layer `l*` with the highest centred accuracy on **dev** worlds.
2. Report centred accuracy at `l*` on **test** worlds with a 95% Wilson interval.

**PASS** if test accuracy at `l*` ≥ **80%** *and* the no-binding control at `l*` ≤ **25%**.
Otherwise **FAIL**. Primary model: **Qwen2.5-0.5B** (base, the model RealTimeReadWrite used).
Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B are reported, not gated.

## What a pass would and would not mean

A pass means: inside one context, the frozen model holds a state for "Anna's brother" that matches its
state for "Ben" closely enough to retrieve Ben among 8 candidates. That is a usable key **in context**.

It does not by itself mean the model path-integrates. The match could come from an induction-style
lookup (the model retrieving "Ben" because "Anna's brother is Ben" is in the context). That is still the
right key for a memory, but it relies on the binding sentence being visible.

The memory question is harder and is **Gate 1**, to be pre-registered separately before its code:
write in chunk 1, clear the KV cache, query by the other route in chunk 2. With no context, "Anna's
brother" cannot resolve to Ben unless the binding itself was written to memory, so Gate 1 needs a
two-hop read: (Anna's brother) → Ben, then Ben → fact. Gate 0 only tells Gate 1 which layer to key on,
or that no layer works and a structural code must be built (the AIbrainV2 line).

## Known limits, stated in advance

* Synthetic, templated English with one relation per sentence; real text is messier.
* Name tokenisation: multi-token names are read at their last token for both routes.
* The prompt ends at REF, so the state has not seen any predicate (no "lives in…" after it).
* One template; template sensitivity is not tested at this gate.
