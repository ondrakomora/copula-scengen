import numpy as np


def compute_pseudoobservations(data: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(data)) / data.size
