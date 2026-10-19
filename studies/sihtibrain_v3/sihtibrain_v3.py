"""Small reciprocal cable and a state-gated, fatigue-steered somatic probe.

All numbers are dimensionless. A cycle is an equilibrium probe followed by a
discrete adaptation update; this is not a model of action potential dynamics.
"""

import numpy as np


PATTERNS = {
    "left": (1., 1., 0., 0.),
    "right": (0., 0., 1., 1.),
    "split": (1., 0., 1., 0.),
}
LEAVES = (2, 3, 5, 6)
ROOTS = (1, 4)
EDGES = ((0, 1, .8), (0, 4, .8), (1, 2, .5), (1, 3, .5),
         (4, 5, .5), (4, 6, .5))


def _operator(root_g):
    h = np.eye(7) * .15
    for (i, j, _), g in zip(EDGES, (*root_g, .5, .5, .5, .5)):
        b = np.eye(7)[i] - np.eye(7)[j]
        h += g * np.outer(b, b)
    return h


def cable(leaves, root_g=(.8, .8), soma_ping=0.):
    """Return (voltages, SPD operator) for four leaf currents and a soma ping."""
    x = np.asarray(leaves, dtype=float)
    g = np.asarray(root_g, dtype=float)
    if x.shape != (4,) or not np.all(np.isfinite(x)):
        raise ValueError("leaves must be four finite currents")
    if g.shape != (2,) or not np.all(np.isfinite(g)) or np.any(g <= 0):
        raise ValueError("root_g must have two positive finite values")
    if not np.isfinite(soma_ping):
        raise ValueError("soma_ping must be finite")
    drive = np.zeros(7)
    drive[list(LEAVES)] = x
    drive[0] = soma_ping
    h = _operator(g)
    return np.linalg.solve(h, drive), h


def sweep(cycles=36, beta=24., retention=.4, bias=1e-6, pulse=1.):
    """Repeated neutral somatic pings with flux-normalized branch adaptation.

    Gates are centered geometrically; adaptation is a fraction of total branch
    flux, so changing pulse amplitude does not change the direction sequence.
    """
    if not isinstance(cycles, int) or cycles < 1 or not (0 <= retention < 1):
        raise ValueError("cycles must be positive and retention in [0,1)")
    if beta < 0 or pulse <= 0 or not np.all(np.isfinite([beta, bias, pulse])):
        raise ValueError("beta, bias and pulse must be finite, beta nonnegative, pulse positive")
    a = np.array([.5 + bias, .5 - bias], dtype=float)
    if np.any(a < 0) or np.any(a > 1):
        raise ValueError("initial adaptation must be in [0,1]")
    gs, fluxes, echoes, adaptations = [], [], [], []
    for _ in range(cycles):
        g = .8 * np.exp(np.clip(-beta * (a - a.mean()), -1.6, 1.6))
        response, _ = cable((0., 0., 0., 0.), g, soma_ping=pulse)
        flux = g * (response[0] - response[list(ROOTS)])
        gs.append(g.copy())
        fluxes.append(flux.copy())
        echoes.append(float(response[0] / pulse))
        adaptations.append(a.copy())
        a = retention * a + (1. - retention) * flux / flux.sum()
    return {"conductance": np.asarray(gs), "flux": np.asarray(fluxes),
            "echo": np.asarray(echoes), "adaptation": np.asarray(adaptations)}


def probe_echo(leaves, root_g, state_gate=True, pulse=1.):
    """Same-operator somatic difference after local, pre-ping state changes W.

    Root voltage under the *common passive cable* supplies the local state.
    The gate is latched before the ping. The echo is thus a susceptibility of
    that latched operator, with no baseline-voltage difference included.
    """
    if pulse <= 0 or not np.isfinite(pulse):
        raise ValueError("pulse must be finite and positive")
    g = np.asarray(root_g, dtype=float)
    if state_gate:
        state, _ = cable(leaves)
        local = 1. / (1. + np.exp(-8. * (state[list(ROOTS)] - 1.9)))
        g = g * (1. + 1.5 * local)
    without, _ = cable(leaves, g)
    with_ping, _ = cable(leaves, g, soma_ping=pulse)
    return float((with_ping[0] - without[0]) / pulse)
