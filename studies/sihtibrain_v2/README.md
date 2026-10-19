# SihtiBrainV2 — A dendrite keeps what the soma cannot see

**Basal evidence → branch-local cable gradients → somatic compression → state-shaped ping → delayed apical gate → response echo → optional persistent conductance write.**

SihtiBrainV2 tests a precise version of the Sihti Light intuition: a dendrite does not simply discard detail. Some distinctions remain locally in voltage gradients and can later be interrogated. A returning somatic event interacts with that local state and a delayed apical signal, possibly changing both what is visible and how the circuit responds in the future.

![Diagram of the two-window circuit](docs/first_echo.svg)

## Run

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python demo.py --output runs/first_echo

Outputs: `metrics.json` plus a six-panel `sihtibrain_v2.png`. CPU only. No ML frameworks required.

## Window 1: purify into distributed gradients

We use a seven-compartment reciprocal cable: soma 0; branch roots 1 and 4; leaves 2,3 under root 1 and 5,6 under root 4. Two equal-strength input configurations:

- **Cluster:** both currents at leaves 2 and 3.
- **Split:** currents at leaves 2 and 5.

For fixed cables, `H(W)=g_leak I+B diag(W) Bᵀ`. Passive voltage dynamics descend `E(v)=½vᵀH(W)v−Iᵀv`: `dv/dt=−∇_v E`. At equilibrium `v*=H(W)⁻¹I`.

Both inputs produce somatic voltage **1.565174859**. Their branch-root residues `r_i=v_root−v_soma` are different, however:

| | Cluster | Split |
|---|---:|---:|
| Soma | 1.565175 | 1.565175 |
| Left-root residue | 0.798201 | 0.146735 |
| Right-root residue | −0.504731 | 0.146735 |

This is *spatial observability*, not automatic lossless compression or image recognition.

## Window 2: probe the operator instead of repeating the read

The somatic pulse amplitude is a deterministic function of the soma's recent state. With exactly equal somatic histories, the sender emits exactly the same ping. This deliberately preserves the first experiment's blind spot.

Independent apical input `a=(a_left,a_right)` arrives with a delay. A local coincidence gate reads branch trace `r`:

    q_i = 1[ping>0] a_i sigmoid(8(r_i+0.25a_i−0.65))

During the returning event, the two soma-root conductances temporarily become

    W_fast,i = W_base,i (1 + 1.6 q_i)

The measured *echo* subtracts the same operator's no-ping baseline:

    echo(W) = [H(W)^(-1)(I+p e_soma)]_soma − [H(W)^(-1)I]_soma

Thus changes in the unpinged baseline are not mislabeled as probe information. Frozen linear cables cannot reveal the difference. A branch-state-dependent cable operator can.

A separate optional *slow* local write is

    W_slow,i ← W_base,i + η q_i max(r_i, 0)

This is an activity-dependent unsupervised write, **not** a supervised error gradient or evidence of perceptual learning. Erasing the old voltage and apical context, then injecting a fresh identical ping, reads the new stored conductance. The time-stepped `live_loop` makes the delayed gate causal (the pulse starts at the soma eight steps before the branch gate can arrive), while retaining the option to freeze the writer.

## Default demonstration: results in model units

| Gate | Observed |
|---|---:|
| Pre-ping soma difference | 0 |
| Fixed-operator ping-echo difference | less than 1e-14 |
| Contextual operator ping-echo difference | −0.11495 |
| Echo difference after erasing branch trace | less than 1e-14 |
| Post-reset read after idealized local write | −0.04817 |
| Post-reset read after delayed time-stepped write | −0.00897 |
| Time-stepped post-reset read, frozen writer | 0 |

There are 19 deterministic tests of mathematical invariants and negative controls. **These results establish what this constructed circuit does under its chosen parameters, not that biological cortical columns work this way.** The phenomenological gate is designed, not learned. It has not been benchmarked on unseen object categories, sequences, or real recordings.

The apical input can be supplied manually to isolate the mechanism or generated from **another column's previous-window voltages** by `context_from_neighbor`. Both compared inputs receive the **same** previous-neighbor context. The current soma does not somehow decode information it never observed.


## Gate 4: later apical evidence trains the basal readout through a returning ping

[Full protocol, results, and limitations](docs/APICAL_CREDIT_GATE4.md).

On the same nine-compartment cable, a **later supervised teaching current enters only at the tuft**, changes the soma's shaped outgoing ping, and reaches basal plastic readout sites as a **fractional two-ping calcium change** multiplied by local eligibility. No target or scalar error is supplied directly to basal synapses; the apical channel still explicitly carries target-minus-prediction information.

Across four random seeds on held-out amplitude-jittered same-branch-versus-split examples, the returning-ping condition reaches **100% accuracy in all four**. No-ping, shuffled-context, disconnected-trunk, erased-eligibility, and expired-eligibility controls remain at the 66.7% majority-class floor. Importantly, **ordinary direct-gradient learning also reaches 100%**; fixed nonlinear branch features were selected in advance, not discovered.

```bash
python demo_apical_credit.py --output runs/apical_credit
```

The script reproduces a six-panel figure and JSON metrics; CI uploads both. This is an abstract two-window feedback/learning gate, **not** a biophysical simulation of tuft plateaus, autonomous axonal spikes, or temporal synaptic calcium traces.


## Gate 5: stopping test — no two-ping advantage over simpler three-factor learning

[Protocol registered before the experiment](docs/GATE5_PREREGISTRATION.md) · [Results and interpretation](docs/GATE5_RESULT.md).

With **the same nine-compartment cable, fixed nonlinear branch features, supervised apical teaching signal, and six independent training seeds**, all active rules reached 100% in-distribution held-out balanced accuracy. Under three shifted domains at 3% calcium read noise, the two-ping log-ratio averaged **62.13%** balanced accuracy, while a simpler graded outgoing-event rule averaged **62.21%**; the event rule had lower binary cross entropy (2.664 vs. 3.028). A same-noisy-calcium fractional-change rule nearly matched the log ratio (62.04%). **Gate 5 did not pass its precommitted distinctive-advantage criterion.**

The mathematical reason is instructive: this toy's calcium proxy is separable, so taking a calcium ratio **cancels the local-voltage factor** and retains only a nonlinear representation of the globally shaped feedback event. The branch-specific update comes from the eligibility trace, not new information in the ratio. This is a negative result for the *extra two-ping algorithm*, not for V2's persistent operator-memory experiment. The next V2 question should be write selectivity and interference in conductance memory.

Run: `python gate5_stopping_test.py --output runs/gate5_stopping_test`. The GitHub CI artifacts contain the four-panel figure and complete per-seed receipts.


## What this adds, and what it does not

Unlike [SihtiBrain](https://github.com/anttiluode/SihtiBrain), which studies passive cables, branch nonlinearities, supervised EqProp nudges and bistable kicks, V2 deliberately studies a **two-window unsupervised read/write loop**. It does not modify the first repo.

Neither the model nor the motivating analogy establishes that dendrites automatically perform object segmentation, that all of cortex performs global gradient descent, or that transformer residual addition equals energy minimization. Cortical Layer 1 input, Martinotti inhibition, and human L2/3 XOR-like responses are motivating biological features, not precise anatomical equivalents of these gates.

See [theory, caveats, and next experiments](docs/THEORY.md).