import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, computed_field
from statsmodels.distributions.empirical_distribution import ECDF

from copula_scengen.modules.copula.copula_sample import CopulaSample
from copula_scengen.modules.functions.inverse_ecdf import inverse_ecdf
from copula_scengen.modules.functions.lower_transformation_bound import lower_transformation_bound
from copula_scengen.modules.utils.margin_type import is_discrete


class CopulaSampleTransformer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame
    copula_sample: CopulaSample

    @computed_field
    @property
    def n_scenarios(self) -> int:
        return self.copula_sample.max_rank

    def transform_discrete_variable(self, data: np.ndarray, rank: int) -> int:
        u1 = (rank - 1) / self.n_scenarios
        u2 = rank / self.n_scenarios

        lower_bound = int(lower_transformation_bound(data=data, arg=u1))
        upper_bound = int(inverse_ecdf(data=data, arg=u2))

        ecdf = ECDF(data)

        return max(range(lower_bound, upper_bound + 1), key=lambda x: ecdf(x) - ecdf(x - 1))

    def transform_continuous_variable(
        self,
        data: np.ndarray,
        rank: int,
        offset: float = 0.0,
    ) -> float:
        return inverse_ecdf(data=data, p=(rank - 0.5) / self.n_scenarios) + offset

    def _calculate_offset(self, data: np.ndarray) -> float:
        computed_mean = (
            sum(inverse_ecdf(data=data, p=(rank - 0.5) / self.n_scenarios) for rank in range(1, self.n_scenarios + 1))
            / self.n_scenarios
        )
        return data.mean() - computed_mean

    def transform(self) -> pd.DataFrame:
        transformed = []
        n_scenarios = self.copula_sample.max_rank

        for scenario_ranks in self.copula_sample.ranks:
            margin_to_value = {}
            for margin_index, rank in enumerate(scenario_ranks):
                margin_data = self.data.iloc[:, margin_index].to_numpy()

                if is_discrete(margin_data):
                    value = self.transform_discrete_variable(margin_data, rank)
                else:
                    offset = self._calculate_offset(margin_data, n_scenarios)
                    value = self.transform_continuous_variable(margin_data, rank, offset)

                margin_to_value[self.data.columns[margin_index]] = value

            transformed.append(margin_to_value)

        return pd.DataFrame(transformed, columns=self.data.columns)
