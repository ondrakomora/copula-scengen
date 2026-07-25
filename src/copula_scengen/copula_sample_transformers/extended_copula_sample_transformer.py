import numpy as np
import pandas as pd

from copula_scengen.copula.copula_sample import CopulaSample
from copula_scengen.copula_sample_transformers import _shared
from copula_scengen.copula_sample_transformers.base import CopulaSampleTransformationStrategy


class ExtendedCopulaSampleTransformer(CopulaSampleTransformationStrategy):
    """Transform copula ranks while preserving discrete-margin extensions."""

    def _extended_inverse_ecdf(self, cumulative: np.ndarray, args: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(cumulative, args, side="left")
        previous_cumulative = np.where(indices == 0, 0.0, cumulative[indices - 1])
        masses = cumulative[indices] - previous_cumulative
        return indices - 1 + (args - previous_cumulative) / masses

    def _discrete_transformations(self, margin_data: np.ndarray, n_scenarios: int) -> np.ndarray:
        value_counts, cumulative = _shared.discrete_distribution(margin_data)
        lower_bounds, upper_bounds = _shared.discrete_bounds(cumulative, n_scenarios)

        ranks = np.arange(1, n_scenarios + 1)
        lower_extended = self._extended_inverse_ecdf(cumulative, (ranks - 1) / n_scenarios)
        upper_extended = self._extended_inverse_ecdf(cumulative, ranks / n_scenarios)

        candidate_values = np.arange(len(value_counts))
        lower_uniform = 1 - candidate_values[None, :] + lower_extended[:, None]
        upper_uniform = 1 - candidate_values[None, :] + upper_extended[:, None]
        overlaps = np.maximum(0.0, np.minimum(1.0, upper_uniform) - np.maximum(0.0, lower_uniform))

        valid = (candidate_values >= lower_bounds[:, None]) & (candidate_values <= upper_bounds[:, None])
        scores = np.where(valid, overlaps * value_counts[None, :], -1.0)
        return scores.argmax(axis=1)

    def transform(self, data: pd.DataFrame, copula_sample: CopulaSample) -> pd.DataFrame:
        """Transform a copula sample into mixed-margin scenarios."""
        return _shared.transform(data, copula_sample, self._discrete_transformations)
