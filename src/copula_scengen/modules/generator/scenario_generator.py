from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from copula_scengen.modules.generator.copula_sample_generator import CopulaSampleGenerator
from copula_scengen.modules.generator.copula_sample_transformer import CopulaSampleTransformer

if TYPE_CHECKING:
    import pandas as pd


class ScenarioGenerator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame

    def generate(self, n_scenarios: int) -> pd.DataFrame:
        copula_sample = CopulaSampleGenerator(data=self.data).generate(n_scenarios=n_scenarios)
        return CopulaSampleTransformer(data=self.data).transform(copula_sample=copula_sample)
