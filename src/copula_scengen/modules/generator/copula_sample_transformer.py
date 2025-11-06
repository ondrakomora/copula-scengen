import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, computed_field

from copula_scengen.modules.copula.copula_sample import CopulaSample
from copula_scengen.modules.functions.discrete_transformation_bounds import discrete_transformation_bounds
from copula_scengen.modules.functions.inverse_ecdf import inverse_ecdf
from copula_scengen.modules.utils.margin_type import is_discrete
from copula_scengen.schemas.margin_type import MarginType


class CopulaSampleTransformer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame

    @computed_field
    @property
    def margin_types(self) -> dict[int, MarginType]:
        margin_types = {}
        for margin_index in range(self.data.shape[1]):
            margin_data = self.data.iloc[:, margin_index].to_numpy()
            if is_discrete(margin_data):
                margin_types[margin_index] = MarginType.DISCRETE
            else:
                margin_types[margin_index] = MarginType.CONTINUOUS
        return margin_types

    def _compute_transformations(self, n_scenarios: int) -> None:
        margin_transformations = np.zeros((self.data.shape[1], n_scenarios), dtype=float)

        ranks = np.arange(1, n_scenarios + 1)
        for margin_index in range(self.data.shape[1]):
            if self.margin_types[margin_index] == MarginType.DISCRETE:
                margin_data = self.data.iloc[:, margin_index].to_numpy()
                value_counts = np.bincount(margin_data.astype(int))
                cumulative = np.cumsum(value_counts) / len(margin_data)

                lower_args = (ranks - 1) / n_scenarios
                upper_args = ranks / n_scenarios

                lower_bounds, upper_bounds = discrete_transformation_bounds(
                    cumulative_relative_counts=cumulative,
                    lower_args=lower_args,
                    upper_args=upper_args,
                )

                max_val = len(value_counts)
                idx = np.arange(max_val)

                valid = (idx >= lower_bounds[:, None]) & (idx <= upper_bounds[:, None])
                counts_2d = np.where(valid, value_counts[None, :], -1)

                margin_transformations[margin_index] = counts_2d.argmax(axis=1)
            else:
                sorted_margin_data = np.sort(self.data.iloc[:, margin_index].to_numpy())

                quantiles = (ranks - 0.5) / n_scenarios
                computed_values = inverse_ecdf(sorted_data=sorted_margin_data, args=quantiles)
                offset = sorted_margin_data.mean() - computed_values.mean()

                margin_transformations[margin_index] = computed_values + offset

        return margin_transformations.T

    def transform(self, copula_sample: CopulaSample) -> pd.DataFrame:
        transformations = self._compute_transformations(n_scenarios=copula_sample.max_rank)

        result = np.take_along_axis(transformations, copula_sample.ranks - 1, axis=0)
        return pd.DataFrame(result, columns=self.data.columns)
