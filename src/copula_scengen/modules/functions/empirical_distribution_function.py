import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field, field_validator


class EmpiricalDistributionFunction(BaseModel):
    """Empirical distribution computed from data samples."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: np.ndarray

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, v: any) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            msg = "Data must be a numpy array"
            raise TypeError(msg)
        if v.ndim != 1:
            msg = "Data must be a 1D array"
            raise ValueError(msg)
        if len(v) == 0:
            msg = "Data must not be empty"
            raise ValueError(msg)
        return np.sort(v)

    @computed_field
    @property
    def n_samples(self) -> int:
        """Number of data points."""
        return len(self.data)

    def cdf(self, x: float) -> float:
        """Return empirical CDF at x."""
        return float(np.searchsorted(self.data, x, side="right") / self.n_samples)

    def icdf(self, p: float) -> float:
        """Return empirical quantile (inverse CDF) for probability p."""
        if not (0 <= p <= 1):
            msg = "Argument must be in [0, 1]"
            raise ValueError(msg)
        idx = int(np.clip(np.ceil(p * self.n_samples) - 1, 0, self.n_samples - 1))
        return float(self.data[idx])
