import numpy as np
import pytest
from sihtistack.ledger import decompose, reconstruct, split_gradient, cosine

def test_telescoping():
    rng = np.random.default_rng(41)
    states = [rng.normal(size=(4, 7)) for _ in range(8)]
    start, bands = decompose(states)
    assert len(bands) == 7
    np.testing.assert_allclose(reconstruct(start, bands), states[-1], atol=1e-14)

def test_final_state_not_unique():
    start = np.zeros(2)
    a = [np.array([1., 1.]), np.array([-1., 0.])]
    b = [np.array([-2., 0.]), np.array([2., 1.])]
    np.testing.assert_array_equal(reconstruct(start, a), reconstruct(start, b))
    assert not np.array_equal(a[0], b[0])

def test_gradient_projection():
    a = np.array([3., 4.])
    g = np.array([2., 0.])
    par, ortho = split_gradient(a, g)
    np.testing.assert_allclose(par + ortho, a)
    np.testing.assert_allclose(par, [3., 0.])
    assert np.dot(ortho, g) == 0.
    assert cosine(a, g) == pytest.approx(0.6)

def test_reject_invalid_shapes():
    with pytest.raises(ValueError):
        decompose([np.zeros(4)])
    with pytest.raises(ValueError):
        decompose([np.zeros(4), np.zeros(5)])
