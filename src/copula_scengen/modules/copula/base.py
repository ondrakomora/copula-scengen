from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd


class Copula(ABC):
    @abstractmethod
    def __call__(self, args: np.ndarray) -> np.ndarray:
        """Evaluate the copula's CDF at the given points."""

    def grid(self, max_rank: int) -> np.ndarray:
        """
        Evaluate the copula on the ``(i / max_rank, r / max_rank)`` lattice.

        Returns a matrix ``M`` of shape ``(max_rank + 1, max_rank + 1)`` with
        ``M[i, r] == self((i / max_rank, r / max_rank))``. The default evaluates the whole
        lattice with a single batched ``__call__``; subclasses may override with a faster
        exact method (e.g. a cumulative histogram for empirical copulas).
        """
        coords = np.arange(max_rank + 1) / max_rank
        first, second = np.meshgrid(coords, coords, indexing="ij")
        args = np.column_stack((first.ravel(), second.ravel()))
        return self(args).reshape(max_rank + 1, max_rank + 1)


class CopulaProvider(ABC):
    @abstractmethod
    def get(self, data: pd.DataFrame, margins: Sequence[int]) -> Copula:
        """Return a copula fit on data.iloc[:, margins]."""
