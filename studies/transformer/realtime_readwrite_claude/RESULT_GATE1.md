# Gate 1 result — FAIL as pre-registered, and the failure is in one place

*Run 2026-10-19 on Antti's machine. Receipt: [results/gate1_qwen2.5-0.5b.json](results/gate1_qwen2.5-0.5b.json).
Decision rule unchanged from [PREREG_GATE1.md](PREREG_GATE1.md).*

Frozen Qwen2.5-0.5B reads 8 bindings and 8 facts once, two delta memories are written from its states,
the context is discarded, and 800 questions ("Where does Anna's brother live?") are answered from the
memories alone. Test worlds use 40 names never seen on dev.

| | test (800 questions) |
|---|---:|
| configuration chosen on dev | layer 14, PCA 64, whitened |
| **two-hop recall from memory, context gone** | **61.8%**, 95% [58.3%, 65.1%] |
| control: no memory | 12.1% |
| control: shuffled bindings | 6.0% |
| **decision** (needs ≥ 80%, controls ≤ 25%) | **FAIL** |

## Reported alongside

| | |
|---|---:|
| hop 1 lands on the right name key | 61.9% |
| one hop, "Where does Ben live?", memory M2 alone | **98.3%** |
| same pipeline without the sparse expansion: two-hop / one-hop | 66.4% / 99.8% |
| naive sentence-end keys (old RealTimeReadWrite style): two-hop / one-hop | 12.8% / 15.2% |
| **the model itself, whole chunk still in context** | **22.9%** |
| exact string table | 100% |

## What it says

* **All of the loss is in hop 1.** Two-hop accuracy (61.8%) equals hop-1 accuracy (61.9%), and the second
  hop works 98.3% of the time. When the memory finds the right person, it finds the right city.
* **Hop 1 is a cue-matching problem.** "Anna's brother" read inside the chunk has to match "Anna's
  brother" read alone in a question. The state at "brother" mixes the anchor name with the relation and
  with whatever came before; when two anchors share a relation, or the left context differs, the match
  slips. This is the weakness flagged before the run.
* **Position is everything.** The same memory keyed at sentence ends, where RealTimeReadWrite read,
  scores chance on both questions. Keyed at the end of the referring phrase, the one-hop question is
  98–100%. That is the RealTimeReadWrite addressing failure, solved for the one-hop case.
* **The expansion did not help here** (61.8% vs 66.4% raw). Unlike Sol's 12-channel phase code, a
  64-component whitened state is already wide enough for 8 keys; the expansion only adds noise to a
  cue that is itself the problem.
* **The memory does what the model cannot.** With everything still in context, the frozen 0.5B model
  answers the two-hop question only 22.9% of the time. The external memory, with the context gone,
  answers 61.8%: 2.7× the model's own ability, from the model's own states.
* **Whitening is essential.** On dev, whitened 64-component keys reach 0.66; unwhitened ones 0.17–0.26.

## Limits

* One model, one template, 8 bindings per memory, one world per memory.
* Hop-1 errors are not broken down by cause in this run (shared relation word vs shared anchor vs
  context); that needs a follow-up diagnostic.
* The configuration is selected on dev from 18 options; the test is a single run.

## Next

The one-hop path already works. Hop 1 needs a cue that carries **which anchor** and **which relation** as
separate parts, instead of a single state at "brother" that blends them. That is TEM's own form:
an address is a start point plus a move. Candidate for a Gate 1b pre-registration: key = sparse
conjunction of the anchor-name state (end of "Anna") and the relation state (end of "brother").
