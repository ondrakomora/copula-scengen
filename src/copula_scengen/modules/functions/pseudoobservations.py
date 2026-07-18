import numpy as np


def compute_pseudoobservations(data: np.ndarray) -> np.ndarray:
    order = np.argsort(data)
    ranks = np.empty(data.size, dtype=np.intp)
    ranks[order] = np.arange(data.size)
    return ranks / data.size
