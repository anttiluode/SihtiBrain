# PREREG — Gate 2b: store the "about to say it" state

*Claude (Opus 5.5), 2026-10-19. Committed before any Gate 2b code and before any run.*

Gate 2 ([RESULT_GATE1B_2.md](RESULT_GATE1B_2.md)) wrote the state the model had at the token "Oslo"
(having just **read** Oslo) into the last position of the question. It steered the 8-way choice
(oracle 48.5%) but the city was never the model's actual next word (top-1 0%). The position being
written to has to **predict** the city, so the stored value should be a state that predicts it.

## Change: the value is a rehearsal state

At write time, for each fact sentence "Ben lives in Oslo.", the memory writer runs the one-sentence
rehearsal

```
Ben lives in Oslo. Ben lives in
```

on the frozen model and keeps the state at the last token, at layer *Lv*: the position where the model,
having just seen the fact, predicts "Oslo". This is done once per fact, at write time, from that fact
alone (no question, no other sentence). Its centred version (dev mean) is stored in M3 under the fact's
name key.

Everything else is fixed from Gate 1b and not re-selected: the same worlds (test seeds 2000–2099,
held-out names, 800 questions) and the same hop-1 / name keys (outer product, layer 17, d 32, no
expansion, fitted on dev seeds 0–49). The write itself is as in Gate 2:
`h_Lv[last] ← h_Lv[last] + α · ‖h_Lv[last]‖ · unit(m)` at the last token of
"Q: Where does Anna's brother live?\nA: Anna's brother lives in", context otherwise empty.

Selection on dev select worlds (seeds 30–49) by memory-write 8-way accuracy over
Lv ∈ {10, 14, 17, 20, 22} × α ∈ {1, 2, 4, 8}. One test run.

## Decision

**PASS** if all hold on the test worlds:

| | criterion |
|---|---|
| (a) oracle write (the true fact's rehearsal state) | 8-way ≥ 80% **and** full-vocabulary top-1 ≥ 50% |
| (b) memory write | 8-way ≥ 0.85 × the two-hop accuracy of the same keys in this run |
| (c) wrong-city write (another fact of the world): rate of the right city | ≤ 25% |
| (d) no write | ≤ 25% |

The top-1 condition in (a) is new and deliberate: Gate 2 showed that steering among 8 candidates is not
the same as the model saying the word.

Reported: memory-write top-1; the rehearsal itself (how often the model's next token after
"Ben lives in Oslo. Ben lives in" is Oslo); the in-context ceiling; the dev selection table.

## What a pass would mean

A frozen 0.5B model reads a text once; the text is discarded; asked "Where does Anna's brother live?",
it says the city as its own next word, through a value its own forward pass produced at write time and
a two-hop external memory addressed by its own states.
