from functools import lru_cache

import numpy as np


@lru_cache
def cdf_steps(sorted_data: tuple[float, ...]) -> np.ndarray:
    arr = np.array(sorted_data)
    _, counts = np.unique(arr, return_counts=True)
    cdf_steps = np.cumsum(counts) / len(arr)
    return np.concatenate(([0.0], cdf_steps))
