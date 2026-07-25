from collections.abc import Sequence

import pandas as pd

from copula_scengen.copula.base import CopulaProvider
from copula_scengen.copula.extended_empirical_copula import ExtendedEmpiricalCopula


class ExtendedEmpiricalCopulaProvider(CopulaProvider):
    def get(self, data: pd.DataFrame, margins: Sequence[int]) -> ExtendedEmpiricalCopula:
        return ExtendedEmpiricalCopula(data=data.iloc[:, list(margins)].to_numpy())
