from functools import lru_cache

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field, field_validator


@lru_cache(maxsize=512)
def evaluate_joint_cdf(data: np.ndarray, x: np.ndarray) -> float:
    return float(np.mean(np.all(data <= x, axis=1)))


class EmpiricalJointDistributionFunction(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: np.ndarray

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, v: any) -> np.ndarray:
        # Must be a numpy array
        if not isinstance(v, np.ndarray):
            msg = "Data must be a numpy.ndarray"
            raise TypeError(msg)

        # Must be 2D: (n_samples, n_dims)
        if v.ndim != 2:  # noqa: PLR2004
            msg = f"Data must be a 2D array, got shape {v.shape}"
            raise ValueError(msg)

        n, d = v.shape
        if n < 1 or d < 1:
            msg = "Data must contain at least one sample and one dimension"
            raise ValueError(msg)

        # Must be finite (no NaN/Inf)
        if not np.isfinite(v).all():
            msg = "Data contains NaN or infinite values"
            raise ValueError(msg)

        return v

    @computed_field
    @property
    def n_samples(self) -> int:
        """Number of data points (samples)."""
        return self.data.shape[0]

    @computed_field
    @property
    def n_dims(self) -> int:
        """Number of dimensions."""
        return self.data.shape[1]

    def evaluate(self, x: np.ndarray) -> float:
        """Compute empirical joint CDF F(x1, ..., xd) = P(X_i ≤ x_i ∀ i)."""
        if x.shape != (self.n_dims,):
            msg = f"Argument must have shape ({self.n_dims},), got {x.shape}"
            raise ValueError(msg)
        return evaluate_joint_cdf(self.data, x)
