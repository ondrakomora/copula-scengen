import numpy as np
import pandas as pd

from copula_scengen.copula.copula_sample import CopulaSample
from copula_scengen.copula_sample_transformers import _shared
from copula_scengen.copula_sample_transformers.base import CopulaSampleTransformationStrategy
from copula_scengen.functions.inverse_ecdf import inverse_ecdf


class EmpiricalCopulaSampleTransformer(CopulaSampleTransformationStrategy):
    """Transform copula ranks with empirical inverse distribution functions."""

    def transform(self, data: pd.DataFrame, copula_sample: CopulaSample) -> pd.DataFrame:
        """Transform a copula sample into empirical marginal scenarios."""
        n_scenarios = copula_sample.max_rank
        ranks = np.arange(1, n_scenarios + 1)
        quantiles = (ranks - 0.5) / n_scenarios
        margin_transformations = np.zeros((data.shape[1], n_scenarios), dtype=float)

        for margin_index in range(data.shape[1]):
            sorted_margin_data = np.sort(data.iloc[:, margin_index].to_numpy())
            margin_transformations[margin_index] = inverse_ecdf(sorted_margin_data, quantiles)

        return _shared.apply_rank_transformations(data, copula_sample, margin_transformations)
