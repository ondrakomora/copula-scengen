import numpy as np
import pytest

from copula_scengen.copula.empirical_copula import EmpiricalCopula


@pytest.fixture
def ec() -> EmpiricalCopula:
    return EmpiricalCopula(data=np.array([[0.1, 0.2], [0.5, 0.8]], dtype=float))


def test_call_wrong_dim(ec: EmpiricalCopula) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        ec(np.array([0.1, 0.2, 0.3], dtype=float))


@pytest.mark.parametrize(
    ("data", "queries", "expected"),
    [
        (
            np.array([[10.0, -5.0], [2.0, 3.0], [7.0, 1.0]]),
            np.array([[1.0, 1.0], [1 / 3, 1.0], [2 / 3, 2 / 3], [1.0, 2 / 3], [2 / 3, 1.0]]),
            np.array([1.0, 1 / 3, 1 / 3, 2 / 3, 2 / 3]),
        ),
        (
            np.array([[3.2, 7.1], [3.2, -1.0], [9.9, 0.0], [0.0, 100.0]]),
            np.array([[1.0, 1.0], [0.75, 0.75], [0.75, 0.25], [1.0, 0.5], [0.25, 1.0]]),
            np.array([1.0, 0.5, 0.25, 0.5, 0.25]),
        ),
        (
            np.array([[5.0, 2.0], [5.0, 8.0], [5.0, 5.0]]),
            np.array([[1.0, 1 / 3], [1.0, 2 / 3], [1.0, 1.0], [0.99, 1.0]]),
            np.array([1 / 3, 2 / 3, 1.0, 0.0]),
        ),
        (
            np.array([[-100.0, 10.0], [-50.0, 20.0], [0.0, 30.0], [50.0, 40.0]]),
            np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [0.5, 0.75], [1.0, 1.0]]),
            np.array([0.25, 0.5, 0.75, 0.5, 1.0]),
        ),
        (
            np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]]),
            np.array([[0.25, 0.25], [0.5, 0.5], [0.5, 0.75], [0.75, 0.75], [1.0, 0.5]]),
            np.array([0.0, 0.0, 0.25, 0.5, 0.5]),
        ),
        (
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
            np.array([[0.5, 0.5], [0.5, 1.0], [1.0, 0.5], [1.0, 1.0]]),
            np.array([0.25, 0.5, 0.5, 1.0]),
        ),
        (
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
            np.array([[1 / 3, 1 / 3, 1 / 3], [2 / 3, 2 / 3, 2 / 3], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
            np.array([1 / 3, 2 / 3, 1.0, 0.0]),
        ),
    ],
)
def test_call_empirical_copula(
    data: np.ndarray,
    queries: np.ndarray,
    expected: np.ndarray,
) -> None:
    ec = EmpiricalCopula(data=data)
    out = ec(queries)
    assert out.shape == expected.shape
    assert np.allclose(out, expected, atol=1e-12)


def test_empirical_copula_is_grounded() -> None:
    copula = EmpiricalCopula(data=np.array([[0.0, 0.0], [1.0, 1.0]]))

    result = copula(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))

    np.testing.assert_array_equal(result, np.zeros(3))
