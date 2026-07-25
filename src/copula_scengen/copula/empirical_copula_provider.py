from collections.abc import Sequence

import pandas as pd

from copula_scengen.copula.base import CopulaProvider
from copula_scengen.copula.empirical_copula import EmpiricalCopula


class EmpiricalCopulaProvider(CopulaProvider):
    def get(self, data: pd.DataFrame, margins: Sequence[int]) -> EmpiricalCopula:
        return EmpiricalCopula(data=data.iloc[:, list(margins)].to_numpy())
