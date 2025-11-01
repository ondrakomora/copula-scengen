from functools import lru_cache

import numpy as np

from copula_scengen.modules.utils.sorted import sorted_array


def inverse_ecdf(data: np.ndarray, arg: float) -> float:
    return _inverse_ecdf(sorted_array(data), arg)


@lru_cache
def _inverse_ecdf(sorted_data: tuple[float, ...], arg: float) -> float:
    if not 0.0 <= arg <= 1.0:
        msg = "Argument must be between 0 and 1."
        raise ValueError(msg)

    n = len(sorted_data)
    rank = min(max(int(np.ceil(arg * n)) - 1, 0), n - 1)
    return float(sorted_data[rank])
