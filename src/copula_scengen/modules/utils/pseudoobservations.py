import numpy as np


def compute_pseudoobservations(data: np.ndarray) -> np.ndarray:
    n = data.shape[0]
    order = np.argsort(data)
    ranks = np.empty(n, float)
    ranks[order] = np.arange(1, n + 1)
    return ranks / (n + 1)
