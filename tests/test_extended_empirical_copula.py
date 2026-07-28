import numpy as np
import pytest

from copula_scengen.copula.empirical_copula import EmpiricalCopula
from copula_scengen.copula.extended_empirical_copula import ExtendedEmpiricalCopula


@pytest.fixture
def eec() -> ExtendedEmpiricalCopula:
    return ExtendedEmpiricalCopula(data=np.array([[0.1, 0.2], [0.5, 0.8]], dtype=float))


def test_call_wrong_dim(eec: ExtendedEmpiricalCopula) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        eec(np.array([0.1, 0.2, 0.3], dtype=float))


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

    queries = np.array([[0.5, 0.5], [1.0, 1.0], [0.0, 0.0]])
    assert np.allclose(eec(queries), np.array([0.25, 1.0, 0.0]), atol=1e-12)


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
    assert np.allclose(out, np.array([0.375, 0.0, 1.0]), atol=1e-12)


def test_call_single_point_1d_broadcast(eec: ExtendedEmpiricalCopula) -> None:
    out = eec(np.array([1.0, 1.0]))
    assert out.shape == (1,)
    assert np.isclose(out[0], 1.0)


def test_discrete_extension_matches_empirical_joint_cdf_at_jump_points() -> None:
    data = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])

    result = ExtendedEmpiricalCopula(data)(np.array([[0.5, 0.5]]))

    assert result == pytest.approx([0.5])


def test_discrete_extension_preserves_negative_dependence_at_jump_points() -> None:
    data = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    queries = np.array([[0.5, 0.5], [0.5, 1.0], [1.0, 0.5], [1.0, 1.0]])

    result = ExtendedEmpiricalCopula(data)(queries)

    np.testing.assert_array_equal(result, np.array([0.0, 0.5, 0.5, 1.0]))


def test_discrete_extension_handles_unbalanced_jump_masses() -> None:
    data = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    queries = np.array([[0.75, 0.25], [0.75, 1.0], [1.0, 0.25], [1.0, 1.0]])

    result = ExtendedEmpiricalCopula(data)(queries)

    np.testing.assert_array_equal(result, np.array([0.25, 0.75, 0.25, 1.0]))


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ([0.5, 0.5], 0.25),
        ([0.75, 0.5], 0.375),
        ([1.0, 0.5], 0.5),
        ([0.75, 0.75], 0.5625),
    ],
)
def test_discrete_extension_interpolates_between_multiple_jump_points(query: list[float], expected: float) -> None:
    data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    result = ExtendedEmpiricalCopula(data)(np.array([query]))

    assert result == pytest.approx([expected])


def test_discrete_extension_is_invariant_to_row_order() -> None:
    data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    reordered = data[[1, 0, 3, 2]]
    query = np.array([[0.5, 0.5]])

    result = ExtendedEmpiricalCopula(data)(query)
    reordered_result = ExtendedEmpiricalCopula(reordered)(query)

    np.testing.assert_array_equal(result, reordered_result)
