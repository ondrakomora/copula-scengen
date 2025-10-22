import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field

from copula_scengen.modules.utils.cdf import get_cdf


class ExtendedEmpiricalQuantileFunction(BaseModel):
    """
    Extended empirical distribution computed from data samples.

    WARNING: Assumes that data are integers starting from 0.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: np.ndarray

    @computed_field
    @property
    def n_samples(self) -> int:
        """Number of data points."""
        return len(self.data)

    @computed_field
    @property
    def cdf(self) -> np.ndarray:
        return get_cdf(self.data)

    def evaluate(self, val: float) -> float:
        if val < 0 or val > 1:
            msg = "Argument must be in interval [0,1]"
            raise ValueError(msg)

        idx = np.searchsorted(self.cdf, val)

        # Exact match
        if idx < len(self.cdf) and self.cdf[idx] == val:
            return val

        lower = self.cdf[idx - 1] if idx > 0 else 0
        upper = self.cdf[idx] if idx < len(self.cdf) else 1

        return val if np.isclose(lower, upper) else idx - 1 + (val - lower) / (upper - lower)
