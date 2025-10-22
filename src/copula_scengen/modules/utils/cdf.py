import numpy as np


def get_cdf(data: np.ndarray) -> np.ndarray:
    unique_vals, counts = np.unique(data, return_counts=True)
    sorted_idx = np.argsort(unique_vals)
    cumulative_counts = np.cumsum(counts[sorted_idx])
    return cumulative_counts / len(data)
