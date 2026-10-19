# Gate 1b and Gate 2 results — both FAIL as pre-registered

*Run 2026-10-19 on Antti's machine. Receipt: [results/gate2_qwen2.5-0.5b.json](results/gate2_qwen2.5-0.5b.json).
Rules unchanged from [PREREG_GATE1B_2.md](PREREG_GATE1B_2.md). 800 test questions, held-out names, chance 12.5%.*

## Gate 1b — start + move: FAIL by 0.9 points

| | |
|---|---:|
| chosen on dev | outer product (TEM conjunction), layer 17, d 32, no expansion |
| **two-hop from memory, context gone** | **79.1%**, 95% [76.2%, 81.8%] |
| hop 1 lands on the right name | 82.4% |
| hop 2 from the stored key | 99.1% |
| control: no memory / shuffled bindings | 13.5% / 2.6% |
| phrase-only key (Gate 1's key) on these worlds | 32.6% (hop 1 33.8%) |
| anchor-only key | 44.8% (hop 1 46.6%) |
| the model itself, chunk in context | 22.8% |

The threshold was 80%; the point estimate is 79.1%, so the decision is FAIL. What the numbers do show:

* **The conjunction is what works.** In worlds built so that neither part alone can resolve a binding,
  the phrase-only key (Gate 1's) falls to 32.6% and the anchor-only key to 44.8%, both below the 50%
  bound for a single part. The outer product of the two reaches 79.1%. The dev selection chose the
  outer product in all of its top five.
* **Harder worlds, better score.** Gate 1 scored 61.8% on worlds where the anchor alone sufficed;
  Gate 1b scores 79.1% on worlds where it cannot.
* **The memory beats the model by 3.5×** (79.1% from memory with the text gone, 22.8% from the model
  with the text present).
* **A dev–test gap of 15 points** (93.8% on dev select worlds, 79.1% on test). Dev and test differ in
  names, so part of what the key maps fit is name identity rather than structure. This is the main
  reason the gate missed.

## Gate 2 — the model speaks from memory: FAIL

Write the city state back into the stream at one layer, at the last token of
"…Anna's brother lives in", and score the model's own next token (8-way / full-vocabulary top-1):

| write | 8-way | top-1 | needed |
|---|---:|---:|---|
| memory | 34.0% | 0.3% | ≥ 67.3% (0.85 × 79.1%) |
| oracle (true city state) | 48.5% | **0.0%** | ≥ 80% |
| wrong city (rate of the right city) | 8.4% | 0.0% | ≤ 25% |
| no write | 13.5% | — | ≤ 25% |

Chosen on dev: layer 6, α = 4, **the edge of the grid** in both dimensions.

* **The write steers, but the model never says the word.** An oracle write lifts the right city from
  13.5% to 48.5% among the 8, and a wrong write pushes it down to 8.4%, so the direction is right. But
  full-vocabulary top-1 is 0%: the city never becomes the model's actual next word.
* **The value was the wrong kind of state.** What was stored is the state *at* the token "Oslo" in
  "Ben lives in Oslo": the representation of having just **read** Oslo. The question position needs
  the representation of being about to **say** Oslo. Those are different directions in the stream;
  adding the first to the second only nudges.
* **Selection hit the grid edge** (earliest layer, largest α), so the best setting of this kind of
  write may lie outside the grid. That does not rescue the decision.

## Next

Gate 2b: store the "about to say it" state instead. At write time, after reading "Ben lives in Oslo.",
run the one-sentence rehearsal "Ben lives in Oslo. Ben lives in" and keep the state at the last token,
the position where the model predicts Oslo. Write that into the question's last position, at later
layers. To be pre-registered before code.
