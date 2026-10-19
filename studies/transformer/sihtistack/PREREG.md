# Gate 1: does a stored middle-layer difference make a frozen model speak?

**Locked default settings before any pretrained-model run (2026-10-19):**
`Qwen/Qwen2.5-0.5B-Instruct`, **layer 14** (zero-based output of decoder
block 14), return strength **1.0**, seeded assignments **41**, **12** names,
candidate cities Oslo, Rome, Paris, Berlin, London, Madrid, Dublin, Tokyo.
Only cities encoded as unique single tokens are scored. Changes to these
defaults after looking at scores are *exploratory*, not confirmatory.

## Arms

1. **none**: ask without supplying a fact.
2. **in_context**: ask with the explicit fact; the model's capability ceiling.
3. **return**: calculate the difference between post-block last-token hidden
   vectors for the fact-containing and fact-free donor; add this difference
   to the fact-free question's last-token state at the same layer.
4. **wrong_city**: same subject, deliberately different city.
5. **random**: fixed-seed Gaussian direction norm-matched to the residue.

Crucially the harness *knows* the subject and selects the donor. It does not
demonstrate automatic memory addressing. Only a last-token vector is written;
earlier token attention K/V states are **not** transferred. The backbone is
frozen. Stored vectors are not modified by a weight-learning rule.

## Readouts

- **Candidate accuracy:** highest next-token logit restricted to accepted
  single-token city candidates (NOT unconstrained generation accuracy).
- **Exact next-token accuracy:** the model's full-vocabulary top token must
  equal the target city token.
- **Margin:** target city logit minus the strongest other city candidate.

## Pre-registered exploratory engineering gate

Only call Gate 1 a preliminary **PASS** if **all** occur:

- In-context candidate accuracy >= 80%.
- Return candidate accuracy >= 60% and return-minus-none >= 25 percentage points.
- Wrong-city and random controls each <= 35% accuracy on the *correct* city.

Always show every arm, even if the ceiling fails. N=12 is a small demonstration,
not a statistically powered discovery.

## Optional oracle gradient diagnostic

On the fact-free query, take the gradient of the *known target-city logit
minus a wrong-city logit* with respect to the post-block hidden state. Compare
that gradient with the stored return direction using dot product, cosine
and parallel/orthogonal decomposition.

**This diagnostic knows the correct answer**. It is not unsupervised learning,
a deployable memory address, or evidence that the network supplies its own
teaching error. It is a sensitivity measurement.

## Failure interpretation

- Context succeeds; return fails: the **tested write channel** cannot
  reproduce the expression. It does not prove that all information vanished.
- Return succeeds; wrong-city redirects: evidence for a causal fact-related
  direction in this controlled setting, not general relational memory.
- Random succeeds equally well: may be generic steering or amplitude effect.
- Context fails: task/prompt/model ceiling inadequate for gate.
- Gradient dot is positive but write fails: a local derivative is not
  sufficient for a finite-amplitude downstream intervention.

The residual-stream sum is an **accounting identity**, not an invertibility
theorem: individual deltas are not identifiable from their final sum.
