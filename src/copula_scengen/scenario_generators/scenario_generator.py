from __future__ import annotations

import pandas as pd

from copula_scengen.copula_sample_generators import CopulaSampleGenerator
from copula_scengen.copula_sample_generators.base import CopulaSampleGenerationStrategy
from copula_scengen.copula_sample_transformers import CopulaSampleTransformer
from copula_scengen.copula_sample_transformers.base import CopulaSampleTransformationStrategy
from copula_scengen.preprocessing import CategoricalEncoder, DataEncoder
from copula_scengen.scenario_generators.base import BaseScenarioGenerator


class ScenarioGenerator(BaseScenarioGenerator):
    def __init__(
        self,
        copula_sample_generation_strategy: CopulaSampleGenerationStrategy | None = None,
        copula_sample_transformation_strategy: CopulaSampleTransformationStrategy | None = None,
        data_encoder: DataEncoder | None = None,
    ) -> None:
        self._copula_sample_generation_strategy = copula_sample_generation_strategy or CopulaSampleGenerator()
        self._copula_sample_transformation_strategy = copula_sample_transformation_strategy or CopulaSampleTransformer()
        self._data_encoder = data_encoder or CategoricalEncoder()

    def set_data_encoder(self, encoder: DataEncoder) -> None:
        if not isinstance(encoder, DataEncoder):
            msg = "encoder must implement DataEncoder"
            raise TypeError(msg)
        self._data_encoder = encoder

    def set_copula_sample_generation_strategy(self, strategy: CopulaSampleGenerationStrategy) -> None:
        if not isinstance(strategy, CopulaSampleGenerationStrategy):
            msg = "strategy must implement CopulaSampleGenerationStrategy"
            raise TypeError(
                msg,
            )
        self._copula_sample_generation_strategy = strategy

    def set_copula_sample_transformation_strategy(self, strategy: CopulaSampleTransformationStrategy) -> None:
        if not isinstance(strategy, CopulaSampleTransformationStrategy):
            msg = "strategy must implement CopulaSampleTransformationStrategy"
            raise TypeError(
                msg,
            )
        self._copula_sample_transformation_strategy = strategy

    def generate(self, data: pd.DataFrame, n_scenarios: int) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            msg = "data must be a pandas DataFrame"
            raise TypeError(msg)
        if not isinstance(n_scenarios, int):
            msg = "n_scenarios must be an int"
            raise TypeError(msg)

        encoded_data, category_mapping = self._data_encoder.encode(data)

        copula_sample = self._copula_sample_generation_strategy.create(data=encoded_data, n_scenarios=n_scenarios)
        result = self._copula_sample_transformation_strategy.transform(data=encoded_data, copula_sample=copula_sample)

        return self._data_encoder.decode(result, category_mapping)
