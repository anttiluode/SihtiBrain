# Gate 5 result: two-ping calcium ratio does not beat simpler local rules

*2026-10-19. The [Gate 5 protocol](GATE5_PREREGISTRATION.md) was committed to the branch before the experiment was run. The result was recorded afterwards, without altering that protocol.*

Reproduce:

    python -m unittest discover -s tests -v
    python gate5_stopping_test.py --output runs/gate5_stopping_test

Outputs: full per-seed/per-domain/per-noise JSON receipts and a four-panel figure. GitHub Actions uploads those generated artifacts. Same nine-compartment cable and fixed nonlinear basal features as Gate 4, six training seeds; learning-rate choices made only by in-domain validation. Four contrast/background domains; three calcium-noise levels; three shifted domains define the preregistered primary score.

## Primary receipts (3% independent multiplicative calcium read noise)

| Update rule | Mean balanced accuracy across three shifts | Shifted binary cross entropy | In-distribution balanced accuracy |
|---|---:|---:|---:|
| **Two-ping log calcium ratio** | **62.13%** | **3.028** | 100% |
| Three-factor fractional calcium change, **same noisy detector measurements** | 62.04% | 3.069 | 100% |
| **Three-factor graded outgoing event** | **62.21%** | **2.664** | 100% |
| Three-factor ternary outgoing event | 61.96% | 4.016 | 100% |
| Direct supervised SGD reference | 62.41% | 2.218 | 100% |
| No-context negative control | 50.00% | 0.665 | 50% |

The no-context model's **ordinary accuracy** is 66.7% because it predicts the majority class; its **balanced accuracy** is only 50%. Its BCE can be low under strong shift because the other trained models become overconfident and wrong, not because no-context predicts the task well.

Three specific out-of-distribution tests (balanced accuracy, 3% calcium noise):

| Method | Low contrast | High contrast | Raised background |
|---|---:|---:|---:|
| Two-ping log calcium ratio | 55.3% | 52.8% | 78.3% |
| Graded event | 55.3% | 52.8% | 78.5% |
| Direct supervised SGD | 55.3% | 52.9% | 79.1% |

At calcium noise 0%, 3% and 10% the principal conclusion is unchanged. The graded-event baseline is intentionally **not affected by calcium-detector noise** because it reads the outgoing pulse instead; this is an unequal sensor advantage, stated explicitly. Crucially, the **fractional-calcium-change baseline using the exact same noisy samples** also closely matches the log-ratio method. Therefore the result is not solely explained by the graded event's sensor advantage.

### Preregistered decision: FAIL

The rule needed to exceed the best non-gradient three-factor baseline by **at least two percentage points** in shifted balanced accuracy *and* attain a lower shifted BCE. Its score was **62.131% vs. 62.207%**, with BCE **3.028 vs. 2.664**. It did not pass.

This is a negative result about **distinctive algorithmic advantage in this toy**, not a refutation of the original V2 operator-memory results, biological calcium-dependent learning, or feedback through apical dendrites.

## Post-result mathematical explanation: the local voltage cancels

The Gate 4/5 proxy for pulse-driven calcium is separable:

    Ca_i(p,v_i) = exp(0.3 v_i) * [exp(k attenuation_i p) - 1]

Therefore, in the absence of measurement noise:

    log[Ca_i(p_later,v_i)/Ca_i(p_early,v_i)]
       = log[(exp(k attenuation_i p_later)-1)/
             (exp(k attenuation_i p_early)-1)].

The multiplicative factor containing **local branch voltage cancels exactly**. The shaped later pulse is a monotone function of one scalar apical teaching residual. The calcium ratio therefore provides **no new branch-specific credit information** beyond the attenuation constant: it merely re-encodes the same globally shaped teaching event. The actual branch selection comes from the fixed local eligibility feature.

This does not rule out a genuinely *nonseparable*, voltage-dependent calcium process. But retrofitting extra channel details now in order to make the two-ping method win would invalidate the stopping-test spirit. A new, independently specified benchmark would be required.

## What carries forward

**Retained:** Gate 1's operator-dependent echo and persistent conductance write, and Gate 4's existence proof of a delayed apical route for a locally gated synaptic change in a toy.

**Not established:** a general learning advantage from two noisy calcium measurements over a simpler three-factor update, or robust generalization of the hand-designed branch features outside their training range. Even the direct-gradient reference performs poorly on the contrast shifts.

**V2 after this gate:** do not chase self-firing pings here (V1 is doing feedback physiology; V3 is doing active probing). A more promising V2 experiment is **operator-memory selectivity under interference**. Write one memory into the conductance operator, then another; compare preservation of the first memory against a matched-byte vector state, direct indexed addressing, and shuffled writers. Probe both with a fixed readout after voltage reset. Pre-register a result that can fail. That tests our most solid hypothesis: memory as a change in the system's future response.

No claims about automatic image segmentation, no-teacher learning, a biological circuit reconstruction, or superiority to conventional neural networks are implied.
