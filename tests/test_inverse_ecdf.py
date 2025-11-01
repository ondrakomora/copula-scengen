import numpy as np
import pytest

from copula_scengen.modules.functions.inverse_ecdf import inverse_ecdf


@pytest.mark.parametrize(
    ("data", "arg", "expected"),
    [
        # Exact boundaries
        ([1, 2, 3], 0.0, 1.0),
        ([1, 2, 3], 1.0, 3.0),
        # Midpoint
        ([1, 2, 3], 0.5, 2.0),
        # Duplicate / discrete emphasis
        ([1, 1, 1, 2, 2, 10], 0.1, 1.0),
        ([1, 1, 1, 2, 2, 10], 0.5, 1.0),
        ([1, 1, 1, 2, 2, 10], 0.51, 2.0),
        ([5, 5, 5, 5], 0.75, 5.0),
        # Additional quantiles
        ([1, 2, 3, 4, 5], 0.25, 2.0),
        ([1, 2, 3, 4, 5], 0.75, 4.0),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9, 9.0),
        # Near boundaries
        ([1, 2, 3, 4, 5], 0.01, 1.0),
        ([1, 2, 3, 4, 5], 0.99, 5.0),
        # Large gaps in data
        ([1, 100, 1000], 0.33, 1.0),
        ([1, 100, 1000], 0.67, 1000.0),
        ([1, 100, 1000], 0.65, 100.0),
        # Negative values
        ([-5, -2, 0, 3, 7], 0.0, -5.0),
        ([-5, -2, 0, 3, 7], 1.0, 7.0),
        ([-5, -2, 0, 3, 7], 0.5, 0.0),
        # Floating point values
        ([1.5, 2.5, 3.5], 0.5, 2.5),
        ([0.1, 0.2, 0.3, 0.4], 0.25, 0.1),
    ],
)
def test_inverse_ecdf(data: list[int | float], arg: float, expected: float) -> None:
    arr = np.array(data, dtype=float)
    out = inverse_ecdf(arr, arg)
    assert out == expected


@pytest.mark.parametrize(
    ("arr", "p", "expected"),
    [
        (np.array([5, 10, 15, 20, 25]), 0.0, 5.0),
        (np.array([5, 10, 15, 20, 25]), 1.0, 25.0),
        (np.array([-100, -50, 0, 50, 100]), 0.0, -100.0),
        (np.array([-100, -50, 0, 50, 100]), 1.0, 100.0),
        (np.array([0.1, 0.5, 0.9, 1.5, 2.3]), 0.0, 0.1),
        (np.array([0.1, 0.5, 0.9, 1.5, 2.3]), 1.0, 2.3),
        (np.array([99, 1, 50, 25, 75]), 0.0, 1.0),
        (np.array([99, 1, 50, 25, 75]), 1.0, 99.0),
        (np.array([3, 3, 7, 7, 7, 15, 15]), 0.0, 3.0),
        (np.array([3, 3, 7, 7, 7, 15, 15]), 1.0, 15.0),
    ],
)
def test_inverse_ecdf_boundaries_return_min_max(arr: list[int | float], p: float, expected: float) -> None:
    assert inverse_ecdf(arr, p) == expected


@pytest.mark.parametrize("arg", [-0.1, 1.1, 2.0, -1.0, np.inf, -np.inf])
def test_inverse_ecdf_invalid_arg(arg: float) -> None:
    arr = np.array([1, 2, 3], dtype=float)
    with pytest.raises(ValueError):  # noqa: PT011
        inverse_ecdf(arr, arg)


def test_inverse_ecdf_single_element() -> None:
    arr = np.array([42.0])
    assert inverse_ecdf(arr, 0.0) == 42.0
    assert inverse_ecdf(arr, 1.0) == 42.0
    assert inverse_ecdf(arr, 0.5) == 42.0


def test_inverse_ecdf_handles_unsorted_input() -> None:
    arr = np.array([3, 1, 2])
    assert inverse_ecdf(arr, 0.0) == 1.0
    assert inverse_ecdf(arr, 1.0) == 3.0
    assert inverse_ecdf(arr, 0.5) == 2.0


def test_inverse_ecdf_all_identical() -> None:
    arr = np.array([7.0, 7.0, 7.0, 7.0, 7.0])
    assert inverse_ecdf(arr, 0.0) == 7.0
    assert inverse_ecdf(arr, 0.3) == 7.0
    assert inverse_ecdf(arr, 0.7) == 7.0
    assert inverse_ecdf(arr, 1.0) == 7.0


def test_inverse_ecdf_empty_array() -> None:
    arr = np.array([])
    # Should raise an error for empty input
    with pytest.raises((ValueError, IndexError)):
        inverse_ecdf(arr, 0.5)
