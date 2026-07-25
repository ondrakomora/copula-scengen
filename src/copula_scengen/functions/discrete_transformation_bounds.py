import numpy as np


def discrete_transformation_bounds(
    cumulative_relative_counts: np.ndarray,
    lower_args: np.ndarray,
    upper_args: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # vectorized searchsorted
    lower_bounds = np.searchsorted(cumulative_relative_counts, lower_args, side="right")
    upper_bounds = np.searchsorted(cumulative_relative_counts, upper_args, side="right")

    # apply same correction logic vectorized
    mask = (~np.isclose(lower_args, 0)) & (~np.isclose(lower_args, 1)) & np.isin(lower_args, cumulative_relative_counts)
    lower_bounds[mask] += 1

    last_idx = len(cumulative_relative_counts) - 1
    lower_bounds = np.minimum(lower_bounds, last_idx)
    upper_bounds = np.minimum(upper_bounds, last_idx)

    return lower_bounds, upper_bounds
