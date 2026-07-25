from typing import Protocol, runtime_checkable

import pandas as pd

from copula_scengen.copula.copula_sample import CopulaSample


@runtime_checkable
class CopulaSampleTransformationStrategy(Protocol):
    def transform(self, data: pd.DataFrame, copula_sample: CopulaSample) -> pd.DataFrame:
        """Transform copula ranks into scenario values."""
        ...
