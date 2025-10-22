from math import ceil

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, computed_field

from copula_scengen.modules.copula.copula_sample import CopulaSample
from copula_scengen.modules.functions.empirical_distribution_function import EmpiricalDistributionFunction
from copula_scengen.modules.functions.extended_empirical_quantile_function import ExtendedEmpiricalQuantileFunction
from copula_scengen.modules.functions.uniform_distribution_function import uniform_distribution_function
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
        edf = EmpiricalDistributionFunction(data=data)

        extended_eqf = ExtendedEmpiricalQuantileFunction(data=data)

        u1 = (rank - 1) / self.n_scenarios
        u2 = rank / self.n_scenarios

        left_bound = extended_eqf.evaluate(u1)
        right_bound = extended_eqf.evaluate(u2)

        n1 = max(0, ceil(left_bound))
        n2 = max(0, ceil(right_bound))

        def score(x: int) -> float:
            return (edf.cdf(x) - edf.cdf(x - 1)) * (
                uniform_distribution_function(right_bound + 1 - x) - uniform_distribution_function(left_bound + 1 - x)
            )

        return max(range(n1, n2 + 1), key=score)

    def transform_continuous_variable(
        self,
        data: np.ndarray,
        rank: int,
        offset: float = 0.0,
    ) -> float:
        edf = EmpiricalDistributionFunction(data=data)

        return edf.icdf((rank - 0.5) / self.n_scenarios) + offset

    def _calculate_offset(self, data: np.ndarray) -> float:
        edf = EmpiricalDistributionFunction(data=data)
        computed_mean = (
            sum(edf.icdf(rank / self.n_scenarios) for rank in range(1, self.n_scenarios + 1)) / self.n_scenarios
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
