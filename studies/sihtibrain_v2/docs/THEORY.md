# From Sihti Light to an observable cortical return path

## Three different meanings of gradient

**Spatial voltage gradient:** an axial current is `g_ij(v_i−v_j)`. The soma integrates a signed balance of axial currents, leak and membrane capacitance. Its sum is not automatically a learning-loss gradient.

**Energy gradient in voltage state:** when the conductances are reciprocal and fixed, a cable obeys `dv/dt=−∇_v E` with `E=½vᵀHv−Iᵀv`. For fixed input, `dE/dt=−||∇_v E||²≤0`. When conductances vary with activity or the system receives pulses, the same fixed-energy monotonicity does not generally hold.

**Learning gradient in weight space:** `∂L/∂W` depends on an objective, feedback and credit assignment. Our local slow write has no global target, and it has not been shown to reduce a task loss. Supervised equilibrium propagation (EqProp) solves a different problem using a signed teacher-provided somatic nudge.

## A lossless residue identity, not automatic compression

For one Euler relaxation step, `x_(k+1)=x_k−η(Hx_k−I)`, define `r_k=η(Hx_k−I)`. It follows exactly that `x_0=x_T+Σ_k r_k`. The residue includes both cable/leak and imposed input terms, and it is a **temporal change in state**. It is not simply a pixel-boundary map. Storing all such residues can cost more space than the original signal.

V2's local signed branch state `v_root−v_soma` is a smaller readout of the distributed voltage differences. The optional slow conductance then changes how later signals propagate. This distinguishes activity-state memory from **operator memory**.

Connections to earlier work: [Sihti](https://github.com/anttiluode/Sihti), [InformationalSpectra](https://github.com/anttiluode/InformationalSpectra), [GraphNN](https://github.com/anttiluode/GraphNN), [LensLuotainTarget](https://github.com/anttiluode/LensLuotainTarget), and [GrowingMatrix](https://github.com/anttiluode/GrowingMatrix). These are architectural connections, not evidence borrowed from those experiments.

## How the cortical diagram can guide the model

Basal dendrites can receive feedforward evidence; soma/axon integration determines output; apical dendrites receive distinct long-range and local contextual input; active dendritic conductances and inhibition affect their interaction. This invites a three-terminal transistor analogy in which context adjusts input-output gain. **The analogy is not an electronic identity.** A biological dendritic tree is not a MOSFET.

Crucial distinction: a backpropagating spike runs through a neuron's own dendritic arbor, but does **not** itself broadcast laterally across all apical tufts in Layer 1. Incoming contextual axons supply independent activity to other tufts. V2 represents that second path as an explicitly supplied delayed apical port, sometimes computed from a neighboring model column's earlier input.

Martinotti-like somatostatin interneurons can inhibit distal dendrites, but cannot be equated to an automatic graph-edge separator between visual objects. The `inhibit` option represents a port shunt, not interneuron physiology.

The XOR-like dendritic effects reported in human layer 2/3 [by Gidon et al. (2020)](https://doi.org/10.1126/science.aax6239) are a **nonlinear input interaction**. They are not mathematically identical to taking a cable Laplacian `Hx`; nor are they proof that all L2/3 neurons isolate image edges.

Biological pointers: [Larkum, Zhu & Sakmann (1999)](https://doi.org/10.1038/18686), [Stuart & Sakmann (1994)](https://doi.org/10.1038/367069a0), and [Brunner & Szabadics (2016)](https://pubmed.ncbi.nlm.nih.gov/?term=Brunner+Szabadics+2016+analogue+modulation+backpropagating+action+potentials) on state-sensitive backpropagation. This model does not reproduce their biological measurements.

## Observability is the fundamental constraint

If two inputs produce the **same entire somatic history** with a fixed linear operator, any sender rule using only that history generates identical pings. The probe echo of the unchanged linear operator must also be identical by superposition. The difference can become observable if:

1. Another local or lateral observer sees it;
2. The probe encounters a state-dependent operator;
3. The circuit stored a difference in its weights/conductances.

V2 tests options 2 and 3 while explicitly supplying the additional apical context. It does not smuggle a cluster label into the ping; the sender's pulse matches for both input patterns.

## Tests that could refute the interesting extrapolation

- Sweep orientations, stimulus strengths, distracting noise and delays with fixed hyperparameters. A left-preferring contextual example is **not** a complete orientation-invariant segmenter.
- Control for parameter count and power: compare with static nonlinear thresholds, frozen cables, apical-only readout, shuffled histories and ordinary gated recurrent units.
- Test memory utility on held-out *sequences* with a hard reset of voltage; show the modified wires preserve the correct distinctions and unrelated states do not drift.
- For image segmentation, connect many receptive-field columns, use past image context without target-mask leakage, and benchmark against ordinary graph diffusion and classical segmentation baselines. This toy has **no image segmentation result**.
- Measure the rank/singular values of the effective input-to-soma observability Jacobian before and after a ping, to find whether context really increases access to hidden distinctions.

**Research question:** can a local feedback event make distributed information selectively readable and writable under strict, causal information constraints, without inserting a teacher-signed error or an oracle context channel?

## The deeper mathematical readout: the spectrum of a disturbance

For the soma's unit probe `e_s`, the frozen-cable probe susceptibility is

    χ(W) = e_sᵀ H(W)⁻¹ e_s.

For a reciprocal edge with incidence vector `b_e`, `H(W)=H_0+Σ_e W_e b_e b_eᵀ`. Differentiating the inverse gives an **exact** identity:

    ∂χ/∂W_e = −(b_eᵀ H(W)⁻¹ e_s)².

The response of the soma to a ping is thus sensitive to the geometry of individual cables, but the sensitivity depends on how strongly the soma's probe reaches each edge. This is the small-circuit form of the InformationalSpectra question: *what modes are observable through this measurement port?* With a fixed operator, soma-null modes remain hidden. Contextual operator changes can mix hidden and visible modes across time windows. This identity is passive linear algebra; the speculative ingredient is whether biological dendrites exploit sufficiently selective, causal conductance modulation.

In the code, `ping_sensitivity_to_cables` implements this identity. The circuit is still too small to infer a neural spectral law or claims about consciousness.
