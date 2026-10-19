"""Bounded local association writes and query-clamped graph relaxation.

The graph's exact equilibrium is a normalized linear fast-weight read.
The common-mode split is Sihti-inspired; it is not the spatial Sihti sieve.
"""
import numpy as np


def vector(x, size=None):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or (size is not None and a.size != size):
        raise ValueError('expected a nonempty vector of the configured size')
    if not np.isfinite(a).all():
        raise ValueError('vectors must contain only finite numbers')
    return a


def split(x):
    """Return the scalar common mode and exact zero-mean residue."""
    a = vector(x)
    scale = float(np.max(np.abs(a)))
    core = float((a / scale).mean() * scale) if scale else 0.
    with np.errstate(over='ignore', invalid='ignore'):
        residue = a - core
    if not np.isfinite(residue).all():
        raise ValueError('split residue exceeds the finite numeric range')
    return core, residue


def cue(x, centered=True):
    """Unit-norm query; a constant centered cue contains no identity."""
    a = vector(x)
    scale = float(np.max(np.abs(a)))
    if not scale:
        return np.zeros_like(a)
    a = a / scale  # normalization is scale invariant; avoid sum/norm overflow
    if centered:
        a = a - a.mean()
    norm = float(np.linalg.norm(a))
    return a / norm if norm > 1e-12 else np.zeros_like(a)


class MemoryBase:
    def __init__(self, dim=64, values=8):
        if (not isinstance(dim, int) or not isinstance(values, int)
                or dim < 1 or values < 1):
            raise ValueError('dim and values must be positive integers')
        self.dim, self.values = dim, values

    def pair(self, x, v):
        x, v = vector(x, self.dim), vector(v, self.values)
        if (x < 0).any() or (x > 1).any():
            raise ValueError('write key must be in [0,1]')
        if (v < 0).any() or (v > 1 + 1e-8).any():
            raise ValueError('observed value must be a probability vector')
        total = float(v.sum())
        if abs(total - 1) > 1e-8:
            raise ValueError('observed value must be a probability vector')
        return x, v / total  # tolerance must never create conductance above one

    def query(self, x, centered=True):
        return cue(vector(x, self.dim), centered=centered)


class ConnectionMemory(MemoryBase):
    """Persistent W plus disposable value-node activity.

    Writes use the observed value, not a reward or backpropagated gradient.
    Rows selected by a key move toward that value. Shared features interfere.
    """
    def __init__(self, dim=64, values=8, eta=0.5, leak=1.0):
        super().__init__(dim, values)
        if not np.isfinite(eta) or not 0 < eta <= 1:
            raise ValueError('eta must be in (0,1]')
        if not np.isfinite(leak) or leak <= 0:
            raise ValueError('leak must be positive and finite')
        self.eta, self.leak = float(eta), float(leak)
        self.weights = np.zeros((dim, values), dtype=np.float32)
        self.activity = np.zeros(values, dtype=np.float64)

    @property
    def state_bytes(self):
        """Persistent numeric state only; transient activity excluded."""
        return self.weights.nbytes

    def write(self, x, v):
        x, v = self.pair(x, v)
        rate = self.eta * x[:, None]
        self.weights[:] = (1 - rate) * self.weights + rate * v[None, :]

    def read(self, x, centered=True):
        q = self.query(x, centered=centered)
        w = self.weights.astype(np.float64)
        self.activity = (w.T @ q) / (self.leak + w.sum(axis=0))
        return self.activity.copy()

    def energy(self, h, q):
        w = self.weights.astype(np.float64)
        return float(0.5 * self.leak * (h @ h)
                     + 0.5 * np.sum(w * (h[None, :] - q[:, None]) ** 2))

    def relax(self, x, steps=80):
        """Diagonally preconditioned energy descent from zero activity.

        h <- h - .25 D^-1 grad(E); no simulated spiking or plastic read.
        Returns activity and energy including the initial state.
        """
        if not isinstance(steps, int) or steps < 1:
            raise ValueError('steps must be a positive integer')
        q = self.query(x)
        w = self.weights.astype(np.float64)
        target = (w.T @ q) / (self.leak + w.sum(axis=0))
        h = np.zeros(self.values)
        energies = [self.energy(h, q)]
        for _ in range(steps):
            h += 0.25 * (target - h)
            energies.append(self.energy(h, q))
        self.activity = h
        return h.copy(), np.asarray(energies)

    def clear_activity(self):
        self.activity.fill(0)

    def erase(self):
        self.weights.fill(0)
        self.clear_activity()
