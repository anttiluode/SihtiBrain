# Gate 5 pre-registration: stop test for the two-ping calcium ratio

**Locked before executing the Gate 5 experiment on October 19, 2026.** This file defines the benchmark and decision rule. Later results must be added in a separate file; do not silently revise this protocol after seeing data.

## Question

Does a two-ping local update `eligibility * log(Ca_later/Ca_early)` deliver a task benefit beyond simpler three-factor updates operating on the **same fixed nonlinear basal features** and receiving comparable delayed apical information?

We do **not** compare against ping-off or feedback-shuffled controls as the decisive baselines: they remove task information and are already known to fail. They remain sanity controls.

## Fixed architecture and task

Use the exact nine-compartment V2 Gate 4 reciprocal cable (`apical_credit.py`), two frozen nonlinear branch features `sigmoid(10*(v_root-2))`, and two trainable readout weights plus bias. Labels: 1 for two active leaves on the same branch (two pairs), 0 for one on each branch (four pairs). Later context supplies the signed supervised residual `target - prediction` at the tuft; no target is used at inference.

Generate amplitudes independently per active leaf. Add small independent background drive on inactive leaves:
- training: active in [0.88,1.12], inactive in [0,0.06], 24 examples per pair (144 total).
- validation: independent seed; same distribution, 24 per pair.
- in-distribution test: independent seed, same distribution, 64 per pair.
- shifted **low-contrast**: active in [0.68,0.92], inactive in [0,0.12], 64 per pair.
- shifted **high-contrast**: active in [1.08,1.35], inactive in [0,0.12], 64 per pair.
- shifted **background**: active in [0.88,1.12], inactive in [0.08,0.25], 64 per pair.

No train/validation/test example shares the same random seed. Inputs are labeled by generating pair; there is no downstream mask or oracle feature. Six training seeds 0..5. Every method receives the same per-seed examples, sample order, initial weights, and supervised later information.

## Read/write rules

All receive local eligibility `phi_r` from the frozen branches. A shaped pulse `p_later` depends on the same tuft residual via the **same** gate-4 transfer `1+0.85*tanh(4*3*alpha*(y-yhat))`, early pulse `p_early=1`, two local calcium channels attenuated by [0.65,1.1] and `Ca(p)=A_r*(exp(1.6*atten_r*p)-1)`.

A. **two_ping_log:** `phi_r * log(Ca_later,r / Ca_early,r)` (candidate).
B. **three_factor_fractional:** `phi_r * ((Ca_later,r-Ca_early,r) / Ca_early,r)`, same two noisy calcium readings; this is the classic fractional-change approximation.
C. **three_factor_event:** `phi_r * (p_later-p_early)`. Receives the same signed outgoing event, without two independent calcium reads. This is a deliberately strong event baseline; noise on the calcium detectors does not affect it.
D. **three_factor_sign:** `phi_r * sign(p_later-p_early)`, with magnitude one except a small dead zone 0.02; a ternary burst/pause/no-change rule.
E. **direct_gradient:** ordinary `phi_r*(target-prediction)` supervised gradient upper/reference bound, explicitly not a local pulse reader.
F. **no_context:** diagnostic negative control.

All methods update bias with the mean of their branch-local signals (or the scalar event). Learning-rate selection is permitted **only by validation binary cross entropy** over the fixed set `[0.02,0.05,0.1,0.2]`. Same grid, steps, epochs and budget per method. Train for 35 epochs; evaluate the selected model separately on all held-out domains. No per-domain tuning.

## Calcium measurement stress

Repeat the whole procedure for independent multiplicative Ca noise σ in `[0,0.03,0.1]` on each pulse's two channels. Same noise draws per dataset presentation and seed for both log and fractional rules, so they are paired. Event rules are explicitly immune to this *particular measurement noise* because they do not access the calcium detectors: this is a comparison of information pathways, not equal sensor hardware.

## Primary decision

Primary score = mean held-out **balanced accuracy** across the three shifted domains and six seeds at 3% Ca noise; secondary = held-out binary cross entropy and accuracy on all domains, plus error bars across seeds.

Call the two-ping ratio **distinctively useful** only if it beats the best non-gradient three-factor baseline by at least **2 percentage points** in shifted-domain mean balanced accuracy **and** has lower shifted-domain mean BCE. Otherwise mark Gate 5 a **negative result for algorithmic advantage under this protocol**, regardless of whether the two-ping rule learns.

Do not confuse this stopping test with a proof of biological credit assignment, a no-teacher system, or a general no-go theorem. Hardware implementation, physiological fidelity and unrelated tasks remain open.
