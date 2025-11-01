import numpy as np
import pytest

from copula_scengen.modules.functions.lower_transformation_bound import lower_transformation_bound


@pytest.mark.parametrize(
    ("data", "arg", "expected"),
    [
        # Unsorted, no repeats
        # sorted -> [0,1,2,3,4], n=5
        (np.array([3, 1, 4, 0, 2]), 0.0, 0.0),
        (np.array([3, 1, 4, 0, 2]), 1.0, 4.0),
        (np.array([3, 1, 4, 0, 2]), 0.33, 1.0),  # normal lookup, no correction
        # Unsorted with repeats
        # sorted -> [0,1,1,2,3], n=5, cdf-steps = 0.0, 0.2, 0.6, 0.8, 1.0
        (np.array([3, 1, 2, 1, 0]), 0.2, 1.0),  # hits cdf-step -> +1
        (np.array([3, 1, 2, 1, 0]), 0.6, 2.0),  # hits cdf-step -> +1
        # More repeats, also unsorted
        (np.array([2, 0, 1, 0, 0]), 0.6, 1.0),
        (np.array([2, 0, 1, 0, 0]), 0.8, 2.0),
        (np.array([2, 0, 1, 0, 0]), 1, 2.0),
        # arg does NOT hit any step → no correction
        (np.array([2, 0, 1, 0, 0]), 0.3, 0.0),
        # Edge: all identical values (still “consecutive” from 0..0)
        (np.zeros(4, dtype=int), 1.0, 0.0),
        (np.zeros(4, dtype=int), 0.5, 0.0),
        (np.zeros(4, dtype=int), 0.1, 0.0),  # not a step → no correction
    ],
)
def test_lower_transformation_bound(data: np.ndarray, arg: float, expected: float) -> None:
    result = lower_transformation_bound(data, arg)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("arg", [-0.1, 1.1])
def test_lower_transformation_bound_invalid_arg(arg: float) -> None:
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError):  # noqa: PT011
        lower_transformation_bound(data, arg)
