import numpy as np
import pytest

from copula_scengen.functions.discrete_transformation_bounds import discrete_transformation_bounds


@pytest.mark.parametrize(
    ("cumulative", "lower_args", "upper_args", "expected_lower", "expected_upper"),
    [
        (
            [0.5, 1.0],
            [0.0, 0.25, 0.5, 0.75],
            [0.25, 0.5, 0.75, 1.0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ),
        (
            [0.2, 0.7, 1.0],
            [0.0, 0.2, 0.4, 0.6, 0.8],
            [0.2, 0.4, 0.6, 0.8, 1.0],
            [0, 1, 1, 1, 2],
            [0, 1, 1, 2, 2],
        ),
        (
            [0.25, 0.5, 1.0],
            [0.0, 0.9, 1.0],
            [0.1, 1.0, 1.0],
            [0, 2, 2],
            [0, 2, 2],
        ),
    ],
)
def test_discrete_transformation_bounds(
    cumulative: list[float],
    lower_args: list[float],
    upper_args: list[float],
    expected_lower: list[int],
    expected_upper: list[int],
) -> None:
    lower, upper = discrete_transformation_bounds(
        np.array(cumulative),
        np.array(lower_args),
        np.array(upper_args),
    )

    np.testing.assert_array_equal(lower, expected_lower)
    np.testing.assert_array_equal(upper, expected_upper)
