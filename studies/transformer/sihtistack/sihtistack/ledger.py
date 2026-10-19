"""Algebraic ledger for streams with additive residual updates.

No claim that individual residues are recoverable from the final sum alone.
"""
import numpy as np

def decompose(states):
    """Return initial state, successive deltas (not spectral bands).

    states[0] is the initial pre-block state; states[k+1] is post-block.
    Do not append final RMSNorm output.
    """
    values = [np.asarray(x, dtype=np.float64) for x in states]
    if len(values) < 2:
        raise ValueError("Need initial hidden state and at least one block output")
    if any(v.shape != values[0].shape for v in values):
        raise ValueError("All hidden states must have identical shape")
    return values[0].copy(), [b - a for a, b in zip(values, values[1:])]

def reconstruct(initial, deltas):
    result = np.asarray(initial, dtype=np.float64).copy()
    for delta in deltas:
        result = result + np.asarray(delta, dtype=np.float64)
    return result

def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a,b) / denominator) if denominator > 0 else 0.0

def split_gradient(vector, gradient):
    """Answer-labelled oracle diagnostic, NOT an inference-time memory method."""
    v = np.asarray(vector, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64)
    if v.shape != g.shape:
        raise ValueError("Mismatched shapes")
    sq = float(np.sum(g*g))
    parallel = g * (float(np.sum(v*g))/sq) if sq > 0 else np.zeros_like(v)
    return parallel, v-parallel
