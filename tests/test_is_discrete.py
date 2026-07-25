import numpy as np

from copula_scengen.functions.is_discrete import is_discrete


def test_identifies_integer_valued_numeric_array() -> None:
    assert is_discrete(np.array([1.0, 2.0, np.nan]))
