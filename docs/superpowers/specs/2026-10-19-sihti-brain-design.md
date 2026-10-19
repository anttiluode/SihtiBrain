# SihtiBrain: branch information and somatic readout

## Intent

Test the intuition that a dendritic tree can turn detailed input into useful local voltage differences, while a soma reads a compressed sum and an axonal spike can send a further signal. The first experiment is a seven-compartment, dimensionless *analogy*, not a biological reconstruction or image segmentation system. It must expose a genuine failure of somatic readout and state exactly which added mechanism changes that failure.

## Model

One soma connects to two symmetric proximal branches, each with two distal compartments. With fixed nonnegative symmetric edge conductances, positive leak, unit capacitance, and externally imposed synaptic currents:

`dv/dt = -L_g v - g_leak v + I(t)` and `J_(i->j) = g_ij (v_i - v_j)`.

For constant input, `E(v) = 1/2 sum_edges g_ij (v_i-v_j)^2 + g_leak/2 sum_i v_i^2 - I dot v` satisfies `dE/dt = -||dv/dt||^2 <= 0`. This is descent in *voltage state*, not a learned loss over synaptic weights. Compare equal-strength inputs on two leaves of one branch with one leaf on each branch. By symmetry, the soma cannot identify these patterns. The branch gradients and local voltages can. An optional phenomenological branch coincidence current (`gain * input_leaf_a * input_leaf_b`) changes the somatic output and can cross an axonal threshold. This is an abstract nonlinear subunit, not an NMDA channel model. A threshold crossing is an output event; we make no automatic claim that it optimizes weights or finds objects.

## Claims and gates

1. Constant-current passive relaxation has nonincreasing electrical energy after the imposed step, with numerical tolerance.
2. The two patterns have nearly identical passive somatic voltage for all sampled times, but different branch voltages/axial currents. Distal detail is available locally although lost in the one-number somatic readout.
3. A branch coincidence mechanism makes clustered input different at the soma while keeping total input strength fixed. A matched linear condition does not.
4. A spike threshold may report the nonlinear distinction; a spike is not itself evidence of credit assignment, oscillation, active sensing, or object segmentation. An identical externally imposed somatic ping in both passive cases changes somatic voltage but does not reveal the hidden branch arrangement, since the system is linear and symmetric.

## Boundaries

Use NumPy and Matplotlib; CPU only. Deterministic command-line demo prints numbers and saves a figure plus JSON in `runs/` (ignored). Test numerical invariants and symmetry using Python `unittest`. No fitted parameters, no image dataset, no claim of experimentally validated biophysics. The imposed ping is a brief somatic current pulse, not a simulated action potential. Source and operator spectra must be distinguished: a visual-image Sihti residue is not automatically a membrane voltage gradient. Follow-up work can test delayed inhibitory feedback, adaptive branch conductances, and locally measured plasticity against matched controls.
