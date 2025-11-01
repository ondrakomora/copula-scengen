from functools import lru_cache

import numpy as np

from copula_scengen.modules.utils.cdf_steps import cdf_steps
from copula_scengen.modules.utils.sorted import sorted_array


def step_function(data: np.ndarray, arg: float) -> tuple[float, float]:
    return _step_function(sorted_array(data), arg)


@lru_cache
def _step_function(sorted_data: tuple[float, ...], arg: float) -> tuple[float, float]:
    if not 0 <= arg <= 1:
        msg = "Argument must be in interval [0,1]"
        raise ValueError(msg)

    if len(sorted_data) == 0:
        msg = "Cannot compute step function on empty data."
        raise ValueError(msg)

    cdf = cdf_steps(sorted_data)
    idx = int(np.searchsorted(cdf, arg))

    if idx < len(cdf) and cdf[idx] == arg:
        return arg, arg

    lower = cdf[idx - 1] if idx > 0 else 0.0
    upper = cdf[idx] if idx < len(cdf) else 1.0
    return float(lower), float(upper)
