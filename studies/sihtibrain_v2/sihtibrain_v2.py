"""SihtiBrainV2: two-window, seven-compartment read/write toy circuit.

Basal currents -> cable voltages and local traces -> somatic ping -> delayed
apical gate -> state-dependent conductance probe -> optional slow write.
Dimensionless, phenomenological; NOT a detailed neuron or AP simulation.
"""
from dataclasses import dataclass
import numpy as np

N = 7
SOMA = 0
ROOTS = (1, 4)
LEAVES = (2, 3, 5, 6)
EDGES = ((0, 1, .8), (0, 4, .8), (1, 2, .5), (1, 3, .5),
         (4, 5, .5), (4, 6, .5))


def sigmoid(x):
    return .5 + .5 * np.tanh(np.asarray(x) / 2)


def operator(root_conductance=(.8, .8), leak=.15):
    """Symmetric positive definite cable operator H(W)."""
    g = list(map(float, root_conductance)) + [e[2] for e in EDGES[2:]]
    h = np.eye(N) * leak
    for (u, v, _), edge_g in zip(EDGES, g):
        b = np.zeros(N)
        b[u], b[v] = 1., -1.
        h += edge_g * np.outer(b, b)
    return h


def currents(leaves):
    x = np.asarray(leaves, dtype=float)
    if x.shape != (4,):
        raise ValueError("expected four basal inputs")
    drive = np.zeros(N)
    drive[list(LEAVES)] = x
    return drive


def cable_energy(v, drive, h):
    return float(.5 * v @ h @ v - drive @ v)


def passive_trajectory(leaves, steps=800, dt=.025, g=(.8, .8)):
    """Fixed reciprocal cable: Euler gradient descent in VOLTAGE state."""
    h, drive = operator(g), currents(leaves)
    v = np.zeros(N)
    vs = [v.copy()]
    energies = [cable_energy(v, drive, h)]
    for _ in range(steps):
        v = v - dt * (h @ v - drive)
        vs.append(v.copy())
        energies.append(cable_energy(v, drive, h))
    return np.stack(vs), np.asarray(energies)


@dataclass(frozen=True)
class Encoded:
    voltage: np.ndarray
    residue: np.ndarray
    leaf_drive: np.ndarray


def encode(leaves, g=(.8, .8)):
    drive = currents(leaves)
    voltage = np.linalg.solve(operator(g), drive)
    residue = voltage[list(ROOTS)] - voltage[SOMA]
    return Encoded(voltage, residue, drive)


def shaped_ping(soma_now, soma_previous=0., base=1.):
    """Sender history shapes pulse amplitude; not a spike waveform model."""
    return float(base * (1 + .20 * np.tanh(
        .5 * soma_now + .25 * (soma_now - soma_previous))))


@dataclass(frozen=True)
class Probe:
    echo: float
    soma_with_ping: float
    soma_without_ping_same_operator: float
    gate: np.ndarray
    fast_g: np.ndarray


def local_gate(encoded, apical, ping, threshold=.65, slope=8.,
               context_gain=.25, trace_override=None):
    """Local trace x independent apical context x a returning ping."""
    a = np.asarray(apical, dtype=float)
    if a.shape != (2,) or np.any(a < 0) or np.any(a > 1):
        raise ValueError("apical must be two values in [0,1]")
    r = encoded.residue if trace_override is None else np.asarray(trace_override, dtype=float)
    q = sigmoid(slope * (r + context_gain * a - threshold))
    return a * float(ping > 0.) * q


def echo_with_operator(g, drive, ping):
    """Same-frozen-operator subtraction, excluding altered baseline voltage."""
    h = operator(g)
    unit = np.zeros(N)
    unit[SOMA] = 1.
    v0 = np.linalg.solve(h, drive)
    v1 = np.linalg.solve(h, drive + ping * unit)
    return float(v1[0] - v0[0]), float(v1[0]), float(v0[0])


def probe(encoded, apical=(1., 0.), ping=1., fast_gain=1.6,
          base_g=(.8, .8), trace_override=None):
    gate = local_gate(encoded, apical, ping, trace_override=trace_override)
    fast_g = np.asarray(base_g, dtype=float) * (1 + fast_gain * gate)
    delta, on, off = echo_with_operator(fast_g, encoded.leaf_drive, ping)
    return Probe(delta, on, off, gate, fast_g)


def write(encoded, apical=(1., 0.), ping=1., eta=.35,
          base_g=(.8, .8), trace_override=None):
    """Local, non-teacher, unsupervised slow write of two root conductances."""
    gate = local_gate(encoded, apical, ping, trace_override=trace_override)
    r = encoded.residue if trace_override is None else np.asarray(trace_override, dtype=float)
    return np.asarray(base_g, dtype=float) + eta * gate * np.maximum(r, 0.)


def reset_and_read(stored_g, diagnostic_ping=1., diagnostic_drive=None):
    """Erase old voltage and context; probe only the stored cable geometry."""
    drive = np.zeros(N) if diagnostic_drive is None else currents(diagnostic_drive)
    return echo_with_operator(stored_g, drive, diagnostic_ping)[0]


def experiment(apical=(1., 0.), ping_base=1., eta=.35, fast_gain=1.6):
    clustered = encode([1, 1, 0, 0])
    split = encode([1, 0, 1, 0])
    pc = shaped_ping(clustered.voltage[0], base=ping_base)
    ps = shaped_ping(split.voltage[0], base=ping_base)
    values = {}
    for name, en, ping in (("cluster", clustered, pc), ("split", split, ps)):
        e0 = echo_with_operator((.8, .8), en.leaf_drive, ping)[0]
        pr = probe(en, apical=apical, ping=ping, fast_gain=fast_gain)
        stored = write(en, apical=apical, ping=ping, eta=eta)
        ctrl = probe(en, apical=apical, ping=ping, fast_gain=fast_gain,
                     trace_override=np.zeros(2))
        values[name] = dict(pre_soma=float(en.voltage[0]),
                            root_voltages=en.voltage[list(ROOTS)].tolist(),
                            residue=en.residue.tolist(), ping=ping,
                            fixed_operator_echo=e0, gated_echo=pr.echo,
                            gated_echo_without_residue=ctrl.echo,
                            gate=pr.gate.tolist(), fast_g=pr.fast_g.tolist(),
                            stored_g=stored.tolist(),
                            post_reset_echo=reset_and_read(stored),
                            no_write_echo=reset_and_read((.8, .8)))
    keys = ("pre_soma", "fixed_operator_echo", "gated_echo",
            "post_reset_echo", "gated_echo_without_residue")
    values["differences"] = {
        k: values["cluster"][k] - values["split"][k] for k in keys}
    return values


def context_from_neighbor(previous_leaves, sharpness=6., inhibit=(0., 0.)):
    """Delayed apical input from another circuit's earlier voltages.

    It cannot know an invisible distinction from the current soma alone.
    A two-port shunt mimics inhibition, not a Martinotti cell biophysics model.
    """
    en = encode(previous_leaves)
    logits = sharpness * en.residue
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    inhibition = np.asarray(inhibit, dtype=float)
    if inhibition.shape != (2,) or np.any(inhibition < 0) or np.any(inhibition > 1):
        raise ValueError("inhibit must be two values in [0,1]")
    return (weights * (1. - inhibition)).clip(0., 1.)


def live_loop(encoded, apical=(1., 0.), ping_base=1., pulse_start=35,
              duration=35, propagation_delay=8, steps=240, dt=.025,
              fast_gain=1.6, writer_eta=0., initial_g=(.8, .8)):
    """Causal Euler loop: gate reaches branches AFTER the soma current begins.

    At each step, v follows the instantaneous cable ODE. Adaptive W generally
    breaks the fixed-energy descent theorem. We do not model an actual axonal
    spike, active wave propagation or ion-channel kinetics.
    """
    v = encoded.voltage.copy()
    slow = np.asarray(initial_g, dtype=float).copy()
    apical = np.asarray(apical, dtype=float)
    ping = shaped_ping(encoded.voltage[0], base=ping_base)
    vs = [v.copy()]
    slow_history = [slow.copy()]
    fast_history = [slow.copy()]
    gates = [np.zeros(2)]
    somatic_current = np.zeros(steps)
    for t in range(steps):
        at_soma = pulse_start <= t < pulse_start + duration
        at_branches = (pulse_start + propagation_delay <= t <
                       pulse_start + propagation_delay + duration)
        somatic_current[t] = ping if at_soma else 0.
        trace = v[list(ROOTS)] - v[SOMA]
        gate = (apical * sigmoid(8. * (trace + .25 * apical - .65))
                if at_branches else np.zeros(2))
        fast = slow * (1 + fast_gain * gate)
        drive = encoded.leaf_drive.copy()
        drive[SOMA] = somatic_current[t]
        v = v + dt * (drive - operator(fast) @ v)
        slow += dt * writer_eta * gate * np.maximum(trace, 0.)
        vs.append(v.copy())
        slow_history.append(slow.copy())
        fast_history.append(fast.copy())
        gates.append(gate.copy())
    return dict(voltage=np.stack(vs), slow_g=np.stack(slow_history),
                fast_g=np.stack(fast_history), gate=np.stack(gates),
                ping_current=somatic_current)


def ping_sensitivity_to_cables(root_conductance=(.8, .8)):
    """Exact derivative of soma ping susceptibility with respect to edge W.

    chi(W) = e_s^T H(W)^(-1) e_s
    d chi / d W_edge = -(b_edge^T H(W)^(-1) e_s)^2.
    """
    unit = np.zeros(N)
    unit[SOMA] = 1.
    response = np.linalg.solve(operator(root_conductance), unit)
    return np.array([-(response[SOMA] - response[i])**2 for i in ROOTS])
