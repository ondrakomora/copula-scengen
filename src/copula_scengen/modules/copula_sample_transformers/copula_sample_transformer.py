import numpy as np
import pandas as pd

from copula_scengen.modules.copula.copula_sample import CopulaSample
from copula_scengen.modules.copula_sample_transformers import _shared
from copula_scengen.modules.copula_sample_transformers.base import CopulaSampleTransformationStrategy


class CopulaSampleTransformer(CopulaSampleTransformationStrategy):
    def _discrete_transformations(self, margin_data: np.ndarray, n_scenarios: int) -> np.ndarray:
        value_counts, cumulative = _shared.discrete_distribution(margin_data)
        lower_bounds, upper_bounds = _shared.discrete_bounds(cumulative, n_scenarios)

        candidate_values = np.arange(len(value_counts))
        valid = (candidate_values >= lower_bounds[:, None]) & (candidate_values <= upper_bounds[:, None])
        counts = np.where(valid, value_counts[None, :], -1)
        return counts.argmax(axis=1)

    def transform(self, data: pd.DataFrame, copula_sample: CopulaSample) -> pd.DataFrame:
        return _shared.transform(data, copula_sample, self._discrete_transformations)
