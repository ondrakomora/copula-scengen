from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class DataEncoder(Protocol):
    def encode(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Validate `data` and encode it, returning the encoded data and a decode mapping."""
        ...

    def decode(self, data: pd.DataFrame, mapping: dict[str, np.ndarray]) -> pd.DataFrame:
        """Reverse `encode`, mapping encoded columns in `data` back to their original values."""
        ...
