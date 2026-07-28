import numpy as np
import pytest

from copula_scengen.functions.is_discrete import is_discrete


def test_identifies_integer_valued_numeric_array() -> None:
    assert is_discrete(np.array([1.0, 2.0, np.nan]))


def test_large_fractional_values_are_not_discrete() -> None:
    assert not is_discrete(np.array([1_000_000.1, 1_000_000.2]))


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([-3.0, -1.0, 0.0], True),
        ([0.1, 0.2], False),
        ([42.0], True),
        ([], True),
        ([np.nan, np.nan], True),
        ([1.0, np.inf], False),
        ([1.0, -np.inf], False),
    ],
)
def test_discrete_classification_edge_cases(values: list[float], expected: bool) -> None:
    assert is_discrete(np.array(values)) is expected
