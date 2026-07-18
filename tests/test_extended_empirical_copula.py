import numpy as np
import pytest

from copula_scengen.modules.copula.empirical_copula import EmpiricalCopula
from copula_scengen.modules.copula.extended_empirical_copula import ExtendedEmpiricalCopula


@pytest.fixture
def eec() -> ExtendedEmpiricalCopula:
    return ExtendedEmpiricalCopula(data=np.array([[0.1, 0.2], [0.5, 0.8]], dtype=float))


def test_call_wrong_dim(eec: ExtendedEmpiricalCopula) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        eec(np.array([0.1, 0.2, 0.3], dtype=float))


def test_call_nan(eec: ExtendedEmpiricalCopula) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        eec(np.array([np.nan, 0.2], dtype=float))


def test_call_inf(eec: ExtendedEmpiricalCopula) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        eec(np.array([np.inf, 0.2], dtype=float))


def test_call_outside_bounds_low(eec: ExtendedEmpiricalCopula) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        eec(np.array([-0.1, 0.5], dtype=float))


def test_call_outside_bounds_high(eec: ExtendedEmpiricalCopula) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        eec(np.array([1.1, 0.5], dtype=float))


def test_call_three_dim(eec: ExtendedEmpiricalCopula) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        eec(np.ones((2, 2, 2)))


def test_all_continuous_matches_empirical_copula() -> None:
    data = np.array([[10.5, -5.25], [2.1, 3.3], [7.7, 1.9]])
    queries = np.array([[1.0, 1.0], [0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.5, 0.74]])

    eec = ExtendedEmpiricalCopula(data=data)
    ec = EmpiricalCopula(data=data)

    assert np.allclose(eec(queries), ec(queries), atol=1e-12)


def test_all_discrete_two_points_uniform() -> None:
    # Two margins, each taking values {0, 1} with equal counts -> jump points at 0, 0.5, 1.
    data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    eec = ExtendedEmpiricalCopula(data=data)

    # At an exact jump point, C* must equal the inner empirical copula (lambda = 0 both sides).
    inner = EmpiricalCopula(data=data)
    queries = np.array([[0.5, 0.5], [1.0, 1.0], [0.0, 0.0]])
    assert np.allclose(eec(queries), inner(queries), atol=1e-12)


def test_discrete_margin_interpolates_between_steps() -> None:
    # Single discrete margin with values {0, 1}, second margin continuous.
    data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    eec = ExtendedEmpiricalCopula(data=data)

    at_lower = eec(np.array([[0.5, 1.0]]))
    at_upper = eec(np.array([[1.0, 1.0]]))
    midpoint = eec(np.array([[0.75, 1.0]]))

    # lambda(0.75) = (0.75 - 0.5) / (1.0 - 0.5) = 0.5 -> value must be the midpoint.
    assert np.allclose(midpoint, 0.5 * (at_lower + at_upper), atol=1e-12)


def test_mixed_margins() -> None:
    data = np.array([[0.0, 10.0], [0.0, 2.0], [1.0, 7.0], [1.0, 4.0]])
    eec = ExtendedEmpiricalCopula(data=data)
    out = eec(np.array([[0.75, 0.5], [0.0, 1.0], [1.0, 1.0]]))
    assert out.shape == (3,)
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_call_single_point_1d_broadcast(eec: ExtendedEmpiricalCopula) -> None:
    out = eec(np.array([1.0, 1.0]))
    assert out.shape == (1,)
    assert np.isclose(out[0], 1.0)
