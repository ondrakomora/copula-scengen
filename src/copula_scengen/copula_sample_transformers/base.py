from abc import ABC, abstractmethod

import pandas as pd

from copula_scengen.copula.copula_sample import CopulaSample


class CopulaSampleTransformationStrategy(ABC):
    @abstractmethod
    def transform(self, data: pd.DataFrame, copula_sample: CopulaSample) -> pd.DataFrame:
        """Transform copula ranks into scenario values."""
