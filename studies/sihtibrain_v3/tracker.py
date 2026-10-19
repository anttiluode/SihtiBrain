"""Causal 16-column moving-bar probe; phase averaging is an explicit control.

Each frame is processed once. A column's outgoing somatic ping is routed to
neighboring columns with its branch-flux fractions. The receiver sees that
ping at the next frame, before reading its new local sensory input.
"""

import numpy as np

from sihtibrain_v3 import cable


def make_movie(seed, steps=20, columns=16, width=3, noise=.35, clutter=.08):
    """A bouncing width-three bar and independently sampled sensory clutter."""
    if columns < width + 4 or width < 1 or steps < 2 or noise < 0 or not 0 <= clutter <= 1:
        raise ValueError("invalid movie dimensions or noise")
    rng = np.random.default_rng(seed)
    left = int(rng.integers(2, columns - width - 1))
    direction = int(rng.choice([-1, 1]))
    mask = np.zeros((steps, columns), dtype=bool)
    for t in range(steps):
        mask[t, left:left + width] = True
        if left + direction < 0 or left + direction > columns - width:
            direction *= -1
        left += direction
    artifacts = rng.binomial(1, clutter, (steps, columns))
    observed = np.clip(mask.astype(float) + artifacts + rng.normal(0., noise, mask.shape),
                       0., 1.5)
    return observed, mask


def local_evidence(frame):
    """Each column reads a three-location patch through the seven-node cable."""
    x = np.asarray(frame, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("frame must be a finite 1D array of at least two columns")
    padded = np.pad(x, 1)
    soma = np.array([cable((padded[i], padded[i + 1], padded[i + 1], padded[i + 2]))[0][0]
                     for i in range(x.size)])
    return 1. / (1. + np.exp(-2. * (soma - 1.5651748593788202)))


def route(signal, phase_flux):
    """For each pulse route left/right to neighbors, retaining pulse order."""
    q = np.asarray(signal, dtype=float)
    flux = np.asarray(phase_flux, dtype=float)
    if q.ndim != 1 or q.size < 2 or not np.all(np.isfinite(q)):
        raise ValueError("signal must have at least two finite columns")
    if flux.ndim != 2 or flux.shape[1] != 2 or np.any(flux < 0) or not np.all(np.isfinite(flux)) or np.any(flux.sum(axis=1) <= 0):
        raise ValueError("phase_flux must be positive two-branch pulse flux")
    fractions = flux / flux.sum(axis=1, keepdims=True)
    received = np.zeros((flux.shape[0], q.size))
    received[:, :-1] += fractions[:, 0, None] * q[None, 1:]
    received[:, 1:] += fractions[:, 1, None] * q[None, :-1]
    return received


def rollout(frames, phase_flux):
    """One-frame-delayed neighbor input plus current basal evidence.

    No current or future target mask enters the state. Both pulses have equal
    energy; this receiver averages them, deliberately discarding phase.
    """
    observations = np.asarray(frames, dtype=float)
    if observations.ndim != 2 or observations.shape[0] < 2 or observations.shape[1] < 2:
        raise ValueError("frames must have at least two times and columns")
    prior = np.zeros_like(observations)
    posterior = np.zeros_like(observations)
    prediction = np.zeros(observations.shape[1])
    for t, frame in enumerate(observations):
        prior[t] = prediction
        posterior[t] = .7 * local_evidence(frame) + .3 * prediction
        # The present output becomes next frame's separately delivered context.
        prediction = .35 * posterior[t] + .65 * route(posterior[t], phase_flux).mean(axis=0)
    return {"prior": prior, "posterior": posterior}


def velocity_forecast(past, now):
    """Simple non-oracle translation baseline using only two observed frames."""
    before = np.asarray(past, dtype=float)
    current = np.asarray(now, dtype=float)
    if before.shape != current.shape or before.ndim != 1:
        raise ValueError("past and now must be same-length 1D observations")
    options = {}
    for step in (-1, 0, 1):
        shifted = np.zeros_like(before)
        if step < 0:
            shifted[:step] = before[-step:]
        elif step > 0:
            shifted[step:] = before[:-step]
        else:
            shifted = before.copy()
        options[step] = float(np.dot(shifted, current))
    direction = max((0, -1, 1), key=lambda step: options[step])
    prediction = np.zeros_like(current)
    if direction < 0:
        prediction[:direction] = current[-direction:]
    elif direction > 0:
        prediction[direction:] = current[:-direction]
    else:
        prediction = current.copy()
    return prediction
