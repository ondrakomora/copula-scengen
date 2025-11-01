from functools import lru_cache

import numpy as np


def sorted_array(data: np.ndarray) -> tuple[float, ...]:
    return _sorted_tuple(tuple(data))


@lru_cache
def _sorted_tuple(data: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(np.sort(np.array(data)))
