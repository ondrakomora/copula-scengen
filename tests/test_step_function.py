import numpy as np
import pytest

from copula_scengen.modules.functions.step_function import step_function


@pytest.mark.parametrize(
    ("data", "arg", "expected"),
    [
        # === Simple distinct ===
        ([1, 2, 3], 1 / 3, (1 / 3, 1 / 3)),  # exact hit
        ([1, 2, 3], 0.2, (0.0, 1 / 3)),  # below first step
        ([1, 2, 3], 0.5, (1 / 3, 2 / 3)),  # middle
        ([1, 2, 3], 0.7, (2 / 3, 1.0)),  # upper region
        ([1, 2, 3], 0.0, (0.0, 0.0)),  # boundary low
        ([1, 2, 3], 1.0, (1.0, 1.0)),  # boundary high
        # === All identical ===
        ([5, 5, 5, 5], 0.0, (0.0, 0.0)),
        ([5, 5, 5, 5], 0.25, (0.0, 1.0)),
        ([5, 5, 5, 5], 1.0, (1.0, 1.0)),  # exact top
        # === Mixed duplicates ===
        ([1, 1, 1, 2], 0.25, (0.0, 0.75)),
        ([1, 1, 1, 2], 0.5, (0.0, 0.75)),
        ([1, 1, 1, 2], 0.75, (0.75, 0.75)),  # exact hit
        ([1, 1, 1, 2], 0.99, (0.75, 1.0)),
        # === Reverse sorted input (sorting must be applied) ===
        ([3, 2, 1], 0.2, (0.0, 1 / 3)),
        ([3, 2, 1], 0.6, (1 / 3, 2 / 3)),
        ([3, 2, 1], 0.9, (2 / 3, 1.0)),
        # === Negative and positive mix ===
        ([-2, -1, 0, 10], 0.1, (0.0, 0.25)),
        ([-2, -1, 0, 10], 0.4, (0.25, 0.5)),
        ([-2, -1, 0, 10], 0.6, (0.5, 0.75)),
        ([-2, -1, 0, 10], 0.9, (0.75, 1.0)),
        # === Non-integer floats ===
        ([0.1, 0.1, 0.4, 0.9, 1.7], 0.2, (0.0, 0.4)),
        ([0.1, 0.1, 0.4, 0.9, 1.7], 0.4, (0.4, 0.4)),  # exact
        ([0.1, 0.1, 0.4, 0.9, 1.7], 0.8, (0.8, 0.8)),
        ([0.1, 0.1, 0.4, 0.9, 1.7], 0.99, (0.8, 1.0)),
    ],
)
def test_step_function_valid(data: list[int | float], arg: float, expected: tuple[float, float]) -> None:
    got = step_function(np.array(data, dtype=float), arg)
    assert np.allclose(got, expected)


@pytest.mark.parametrize("arg", [-1e-9, -5, 1.0001, 2])
def test_step_function_invalid_arg(arg: float) -> None:
    data = np.array([1, 2, 3], dtype=float)
    with pytest.raises(ValueError, match="Argument must be in interval"):
        step_function(data, arg)


def test_step_function_empty() -> None:
    data = np.array([], dtype=float)
    with pytest.raises(ValueError):  # noqa: PT011
        step_function(data, 0.5)


def test_step_function_single_value() -> None:
    data = np.array([42.0], dtype=float)
    # CDF is [0.0, 1.0]
    assert step_function(data, 0.0) == (0.0, 0.0)
    assert step_function(data, 0.5) == (0.0, 1.0)
    assert step_function(data, 1.0) == (1.0, 1.0)  # exact hit


def test_step_function_sorted_vs_unsorted_same_result() -> None:
    data1 = np.array([10, 1, 5, 5, 3], dtype=float)
    data2 = np.sort(data1)
    result1 = step_function(data1, 0.6)
    result2 = step_function(data2, 0.6)
    assert np.allclose(result1, result2)
