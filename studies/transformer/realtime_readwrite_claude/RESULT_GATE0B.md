# Gate 0b result — the address exists, in a subspace (post-result diagnostics)

*Run 2026-10-19 on Antti's machine. Receipt: [results/gate0b_qwen2.5-0.5b.json](results/gate0b_qwen2.5-0.5b.json).
Written after Gate 0 failed; it explains that failure and cannot reverse it.*

## A. The model does resolve the two-hop

Prompt `…Now consider Anna's brother. Anna's brother is` → next token:

| | |
|---|---|
| top-1 over the whole vocabulary is Ben | **77.0%** |
| 8-way choice among the world's targets | **81.7%** |
| same, with the relation sentences removed | 11.4% (chance 12.5%) |

So reading 1 of RESULT_GATE0.md ("too small for the two-hop") is out.

## B and C. Where the answer lives

Accuracy of retrieving Ben's direct-route state among the world's 8 targets, held-out test worlds:

| layer | B: cosine at "brother" | B: cosine at "is" | **C: linear map at "brother"** | **C: linear map at "is"** | C control (no relation sentences) |
|---:|---:|---:|---:|---:|---:|
| 10 | 21.7% | 15.8% | 37.5% | 65.0% | 11–12% |
| 14 | 41.4% | 26.2% | 72.2% | 77.3% | 13–14% |
| 17 | 26.2% | 18.2% | **87.4%** | 96.3% | 14–15% |
| 18 | 24.0% | 16.7% | 84.6% | **96.6%** | 13–14% |
| 24 | 22.8% | 16.0% | 69.0% | 90.6% | 11–13% |

The linear map is ridge regression fitted on dev worlds only (λ chosen on a dev split), mapping the
query state to the target's state, and scored on held-out test worlds. The no-binding control, the same
pipeline on contexts without relation sentences, stays at 11–15% at every layer, so the map is not
inventing the answer.

* **Reading 3 holds.** At "brother", which is not where the model has to say anything, Ben is linearly
  recoverable 87% of the time at layer 17, while plain cosine there gives 26%. The referent is present,
  in a subspace that whole-state cosine drowns in other directions (token identity, position). That is
  why Gate 0 failed: the address exists, but not in a form a cosine on raw states can use.
* **Reading 2 is not needed.** At "is", where the model must produce the name, recovery reaches 96.6%,
  higher than the model's own 8-way answer (81.7%). But plain cosine at "is" is *worse* than at
  "brother": the producing position holds the answer only in the mapped subspace too.
* **Where:** the subspace forms at layers 10–14 and is cleanest at 17–19, *later* than the raw-cosine
  peak (14). The final layers lose some of it.

## Limits

* The map is fitted, so this is a statement about **linear decodability**, not about a state the model
  uses as an address. That is enough for a memory key, since any key projection is a learned map.
* Dev and test worlds draw from the same 100 names. The map may partly learn name identities rather than
  an abstract binding code. A held-out-names test is needed before calling it a binding ID.
* At late layers the "is" state is partly the next-token prediction for "Ben", so the 96.6% there may be
  closer to "which name comes next" than to an abstract address. The 87% at "brother" carries no such
  prediction and is the cleaner evidence.
* One model (0.5B), one template, post-result analysis.

## What this means for Gate 1

The key for the live memory should be the layer-17 state at the end of a referring phrase, passed
through a learned projection, not the raw state, and then sharpened by a sparse expansion as in
[RESULT_SHARP.md](RESULT_SHARP.md). Gate 1 will be pre-registered with held-out names and with the
two-hop read that the cleared context makes necessary (PREREG.md, last section).
