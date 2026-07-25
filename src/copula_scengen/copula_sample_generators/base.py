from typing import Protocol, runtime_checkable

import pandas as pd

from copula_scengen.copula.copula_sample import CopulaSample


@runtime_checkable
class CopulaSampleGenerationStrategy(Protocol):
    def create(self, data: pd.DataFrame, n_scenarios: int) -> CopulaSample:
        """Create a rank-based copula sample."""
        ...
