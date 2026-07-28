from collections.abc import Callable

import numpy as np
import pandas as pd

from copula_scengen.copula.copula_sample import CopulaSample
from copula_scengen.functions.discrete_transformation_bounds import discrete_transformation_bounds
from copula_scengen.functions.inverse_ecdf import inverse_ecdf
from copula_scengen.functions.is_discrete import is_discrete


def continuous_transformations(margin_data: np.ndarray, n_scenarios: int) -> np.ndarray:
    sorted_margin_data = np.sort(margin_data)
    ranks = np.arange(1, n_scenarios + 1)
    quantiles = (ranks - 0.5) / n_scenarios
    computed_values = inverse_ecdf(sorted_data=sorted_margin_data, args=quantiles)
    offset = sorted_margin_data.mean() - computed_values.mean()
    return computed_values + offset


def discrete_distribution(margin_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    integer_data = margin_data.astype(int)
    support = np.unique(integer_data)
    if support.size == 0 or not np.array_equal(support, np.arange(support.size)):
        msg = "Discrete margins must contain contiguous integer values starting at zero"
        raise ValueError(msg)

    value_counts = np.bincount(integer_data)
    cumulative = np.cumsum(value_counts) / len(margin_data)
    return value_counts, cumulative


def discrete_bounds(cumulative: np.ndarray, n_scenarios: int) -> tuple[np.ndarray, np.ndarray]:
    ranks = np.arange(1, n_scenarios + 1)
    return discrete_transformation_bounds(
        cumulative_relative_counts=cumulative,
        lower_args=(ranks - 1) / n_scenarios,
        upper_args=ranks / n_scenarios,
    )


def transform(
    data: pd.DataFrame,
    copula_sample: CopulaSample,
    discrete_selector: Callable[[np.ndarray, int], np.ndarray],
) -> pd.DataFrame:
    n_scenarios = copula_sample.max_rank
    margin_transformations = np.zeros((data.shape[1], n_scenarios), dtype=float)

    for margin_index in range(data.shape[1]):
        margin_data = data.iloc[:, margin_index].to_numpy()
        if is_discrete(margin_data):
            margin_transformations[margin_index] = discrete_selector(margin_data, n_scenarios)
        else:
            margin_transformations[margin_index] = continuous_transformations(margin_data, n_scenarios)

    return apply_rank_transformations(data, copula_sample, margin_transformations)


def apply_rank_transformations(
    data: pd.DataFrame, copula_sample: CopulaSample, margin_transformations: np.ndarray
) -> pd.DataFrame:
    result = np.take_along_axis(margin_transformations.T, copula_sample.ranks - 1, axis=0)
    return pd.DataFrame(result, columns=data.columns)
