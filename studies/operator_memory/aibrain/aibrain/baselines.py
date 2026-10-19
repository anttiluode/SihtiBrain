"""Ordinary association-memory controls with explicit storage budgets."""
import numpy as np
from .memory import MemoryBase


class HebbianMemory(MemoryBase):
    """Additive linear fast weights, with the same float32 matrix budget."""
    def __init__(self, dim=64, values=8):
        super().__init__(dim, values)
        self.weights = np.zeros((dim, values), dtype=np.float32)

    @property
    def state_bytes(self):
        return self.weights.nbytes

    def write(self, x, v):
        x, v = self.pair(x, v)
        self.weights[:] += np.outer(self.query(x), v).astype(np.float32)

    def read(self, x):
        return self.weights.astype(np.float64).T @ self.query(x)


class DeltaMemory(HebbianMemory):
    """Unit-rate local delta-rule fast weights, no backpropagation.

    The observed value supplies an explicit prediction target at write time.
    """
    def write(self, x, v):
        x, v = self.pair(x, v)
        k = self.query(x)
        prediction = self.weights.astype(np.float64).T @ k
        self.weights[:] += np.outer(k, v - prediction).astype(np.float32)


class HistoryAttention(MemoryBase):
    """Full event cache, temperature 12 cosine softmax, no recency oracle.

    Key and value vectors are stored in float32; Python container overhead
    is excluded from state_bytes just as object overhead is for other methods.
    """
    def __init__(self, dim=64, values=8):
        super().__init__(dim, values)
        self.keys, self.targets = [], []

    @property
    def state_bytes(self):
        return sum(a.nbytes for a in self.keys + self.targets)

    def write(self, x, v):
        x, v = self.pair(x, v)
        self.keys.append(self.query(x).astype(np.float32))
        self.targets.append(v.astype(np.float32).copy())

    def read(self, x):
        q = self.query(x)
        if not self.keys or not np.any(q):
            return np.zeros(self.values)
        logits = 12 * (np.asarray(self.keys, dtype=np.float64) @ q)
        weights = np.exp(logits - logits.max())
        return weights @ np.asarray(self.targets) / weights.sum()
