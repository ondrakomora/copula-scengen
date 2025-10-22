import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field, field_validator


class EmpiricalCopula(BaseModel):
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

    @computed_field
    @property
    def pseudo_observations(self) -> np.ndarray:
        grouped = []
        for j in range(self.n_dims):
            col = self.data[:, j]
            # unique values and counts
            unique_vals, counts = np.unique(col, return_counts=True)
            # sort unique values
            sorted_idx = np.argsort(unique_vals)
            counts = counts[sorted_idx]
            # normalize by number of unique values
            normalized = counts / len(col)
            grouped.append(normalized)
        return np.array(grouped, dtype=float).T

    def evaluate(self, u: np.ndarray) -> float:
        if u.shape != (self.n_dims,):
            msg = f"Argument must have shape ({self.n_dims},), got {u.shape}"
            raise ValueError(msg)
        if not np.all((u >= 0) & (u <= 1)):
            msg = "All elements of argument must be in [0, 1]"
            raise ValueError(msg)

        return np.mean(np.all(u >= self.pseudo_observations, axis=1))
