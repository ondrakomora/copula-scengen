from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd


class Copula(ABC):
    @abstractmethod
    def __call__(self, args: np.ndarray) -> np.ndarray:
        """Evaluate the copula's CDF at the given points."""


class CopulaProvider(ABC):
    @abstractmethod
    def get(self, data: pd.DataFrame, margins: Sequence[int]) -> Copula:
        """Return a copula fit on data.iloc[:, margins]."""
