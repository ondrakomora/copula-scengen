from abc import ABC, abstractmethod

import pandas as pd

from copula_scengen.copula.copula_sample import CopulaSample


class CopulaSampleGenerationStrategy(ABC):
    @abstractmethod
    def create(self, data: pd.DataFrame, n_scenarios: int) -> CopulaSample:
        """Create a rank-based copula sample."""
