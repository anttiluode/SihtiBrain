"""Dependency-light algebra demo: additive bookkeeping is not invertible memory."""
import numpy as np
from .ledger import decompose, reconstruct

def main():
    x = np.array([1., 2.])
    stages = [x, x + np.array([3., -2.]), x + np.array([2., 2.])]
    initial, residues = decompose(stages)
    err = np.max(np.abs(reconstruct(initial, residues) - stages[-1]))
    print(f"Ledger reconstruction error: {err:.2e}")
    a = [np.array([5., 0.]), np.array([-4., 2.])]
    b = [np.array([2., 3.]), np.array([-1., -1.])]
    same = np.allclose(reconstruct(x,a), reconstruct(x,b))
    print(f"Two different residue histories have identical endpoints: {same}")
    assert err < 1e-12 and same

if __name__ == "__main__":
    main()
