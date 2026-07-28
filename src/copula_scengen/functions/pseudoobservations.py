import numpy as np


def compute_pseudoobservations(data: np.ndarray) -> np.ndarray:
    """Convert one-dimensional observations to upper empirical ranks."""
    if data.size == 0:
        return np.empty(0, dtype=float)

    sorted_data = np.sort(data)
    return np.searchsorted(sorted_data, data, side="right") / data.size
