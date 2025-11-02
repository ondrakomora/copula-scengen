import numpy as np


def inverse_ecdf(sorted_data: np.ndarray, args: np.ndarray) -> np.ndarray:
    # args: array of floats in [0, 1]

    if np.any((args < 0.0) | (args > 1.0)):
        msg = "Argument must be between 0 and 1."
        raise ValueError(msg)

    n = sorted_data.size
    # comment: compute ranks in one shot, clamp to [0, n-1]
    ranks = np.clip(np.ceil(args * n).astype(int) - 1, 0, n - 1)

    return sorted_data[ranks]
