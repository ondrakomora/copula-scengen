import numpy as np
import pytest

from copula_scengen.modules.copula.empirical_copula import EmpiricalCopula


@pytest.fixture
def ec() -> EmpiricalCopula:
    return EmpiricalCopula(data=np.array([[0.1, 0.2], [0.5, 0.8]], dtype=float))


def test_call_wrong_dim(ec: EmpiricalCopula) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        ec(np.array([0.1, 0.2, 0.3], dtype=float))


def test_call_nan(ec: EmpiricalCopula) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        ec(np.array([np.nan, 0.2], dtype=float))


def test_call_inf(ec: EmpiricalCopula) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        ec(np.array([np.inf, 0.2], dtype=float))


def test_call_outside_bounds_low(ec: EmpiricalCopula) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        ec(np.array([-0.1, 0.5], dtype=float))


def test_call_outside_bounds_high(ec: EmpiricalCopula) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        ec(np.array([1.1, 0.5], dtype=float))


def test_call_three_dim(ec: EmpiricalCopula) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        ec(np.ones((2, 2, 2)))


@pytest.mark.parametrize(
    ("data", "queries", "expected"),
    [
        (
            np.array([[10.0, -5.0], [2.0, 3.0], [7.0, 1.0]]),
            np.array([[1.0, 1.0], [0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.5, 0.74]]),
            np.array([1.0, 0.0, 0.0, 1 / 3, 2 / 3]),
        ),
        (
            np.array([[-100.0, 50.0], [-50.0, 10.0], [0.0, 0.0]]),
            np.array([[1.0, 1.0], [0.5, 0.0], [0.2, 0.2]]),
            np.array([1.0, 0.0, 0.0]),
        ),
        (
            np.array([[1000.0, 1000.0], [0.1, 0.1], [500.0, 300.0]]),
            np.array([[1.0, 1.0], [0.4, 0.4], [0.2, 0.9]]),
            np.array([1.0, 2 / 3, 1 / 3]),
        ),
        (
            np.array([[3.2, 7.1], [3.2, -1.0], [9.9, 0.0], [0.0, 100.0]]),
            np.array([[1.0, 1.0], [0.1, 0.9], [0.5, 0.5]]),
            np.array([1.0, 1 / 4, 2 / 4]),
        ),
        (
            np.array([[5.0, 2.0], [5.0, 8.0], [5.0, 5.0]]),
            np.array([[0.6, 0.6], [0.2, 0.2], [1.0, 0.0]]),
            np.array([1 / 3, 1 / 3, 1 / 3]),
        ),
        (
            np.array([[9.0, -3.0], [4.0, -3.0], [1.0, -3.0]]),
            np.array([[0.7, 0.7], [0.3, 0.3], [1.0, 0.0]]),
            np.array([1.0, 0.0, 1 / 3]),
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
