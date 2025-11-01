import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field, field_validator

from copula_scengen.modules.utils.pseudoobservations import compute_pseudoobservations


class EmpiricalCopula(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: np.ndarray

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, v: any) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            msg = "Data must be a numpy.ndarray"
            raise TypeError(msg)
        if v.ndim != 2:  # noqa: PLR2004
            msg = f"Data must be a 2D array, got shape {v.shape}"
            raise ValueError(msg)
        n, d = v.shape
        if n < 1 or d < 1:
            msg = "Data must contain at least one sample and one dimension"
            raise ValueError(msg)
        if not np.isfinite(v).all():
            msg = "Data contains NaN or infinite values"
            raise ValueError(msg)
        return v

    @computed_field
    @property
    def pseudo_observations(self) -> np.ndarray:
        per_margin = [compute_pseudoobservations(self.data[:, j]) for j in range(self.data.shape[1])]
        return np.vstack(per_margin).T.astype(float)

    def __call__(self, args: np.ndarray) -> np.ndarray:
        # allow (d,) -> (1, d)
        if args.ndim == 1:
            args = args[None, :]

        d = args.shape[1]
        if d != self.data.shape[1]:
            msg = f"Each argument must have dimension {self.data.shape[1]}, got {d}"
            raise ValueError(msg)

        if len(args.shape) != 2:  # noqa: PLR2004
            msg = "Arguments must be a 2D array"
            raise ValueError(msg)

        if not np.isfinite(args).all():
            msg = "Arguments contain NaN or infinite values"
            raise ValueError(msg)

        if not np.all((args >= 0) & (args <= 1)):
            msg = "All elements of argument must be in [0, 1]"
            raise ValueError(msg)

        return np.mean((args[:, None, :] >= self.pseudo_observations[None, :, :]).all(axis=2), axis=1)
