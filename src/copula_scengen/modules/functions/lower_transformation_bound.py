from functools import lru_cache

import numpy as np

from copula_scengen.modules.utils.cdf_steps import cdf_steps
from copula_scengen.modules.utils.sorted import sorted_array


def lower_transformation_bound(data: np.ndarray, arg: float) -> float:
    return _lower_transformation_bound(sorted_array(data), arg)


@lru_cache
def _lower_transformation_bound(sorted_data: tuple[float, ...], arg: float) -> float:
    if not 0.0 <= arg <= 1.0:
        msg = "Argument must be between 0 and 1."
        raise ValueError(msg)

    n = len(sorted_data)

    rank = min(max(int(np.ceil(arg * n)) - 1, 0), n - 1)

    if np.isclose(arg, 0.0) or np.isclose(arg, 1.0):
        correction = 0
    elif np.any(np.isclose(arg, cdf_steps(sorted_data))):
        correction = 1
    else:
        correction = 0

    return float(sorted_data[rank]) + correction
