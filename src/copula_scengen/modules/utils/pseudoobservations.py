from functools import lru_cache

import numpy as np


def compute_pseudoobservations(data: np.ndarray) -> np.ndarray:
    return _compute_pseudoobservations(tuple(data))


@lru_cache
def _compute_pseudoobservations(data: tuple[float, ...]) -> np.ndarray:
    data_np = np.array(data)
    return np.argsort(np.argsort(data_np)) / data_np.size
