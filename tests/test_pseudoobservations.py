import numpy as np
import pytest

from copula_scengen.modules.utils.pseudoobservations import compute_pseudoobservations


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (np.array([5, 2, 7, 3]), np.array([2, 0, 3, 1]) / 4),
        (np.array([1, 1, 2, 2]), np.array([0, 1, 2, 3]) / 4),
        (np.array([1, 2, 3]), np.array([0, 1, 2]) / 3),
        (np.array([3, 2, 1]), np.array([2, 1, 0]) / 3),
        (np.array([-1.5, 0.0, 10.2]), np.array([0, 1, 2]) / 3),
        (np.array([1e9, 1e8, 1e7]), np.array([2, 1, 0]) / 3),
        (np.array([4, 4, 4]), np.array([0, 1, 2]) / 3),
        (np.array([0, -1, -1, 10]), np.array([2, 0, 1, 3]) / 4),
        (np.array([1.1, 1.1, 1.2]), np.array([0, 1, 2]) / 3),
    ],
)
def test_compute_pseudoobservations(data: np.ndarray, expected: np.ndarray) -> None:
    out: np.ndarray = compute_pseudoobservations(data)
    assert np.allclose(out, expected)
    assert out.shape == data.shape


def test_empty() -> None:
    data: np.ndarray = np.array([])
    out: np.ndarray = compute_pseudoobservations(data)
    assert out.size == 0
