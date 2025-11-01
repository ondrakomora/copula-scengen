import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field, field_validator

from copula_scengen.modules.copula.empirical_copula import EmpiricalCopula
from copula_scengen.modules.functions.step_function import step_function
from copula_scengen.modules.utils.margin_type import is_discrete
from copula_scengen.schemas.margin_type import MarginType


class EmpiricalExtensionCopula2D(BaseModel):
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

        if d != 2:  # noqa: PLR2004
            msg = "Data must be 2-dimensional for EmpiricalExtensionCopula2D"
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
    def first_margin_type(self) -> MarginType:
        return MarginType.DISCRETE if is_discrete(self.data[0]) else MarginType.CONTINUOUS

    @computed_field
    @property
    def second_margin_type(self) -> MarginType:
        return MarginType.DISCRETE if is_discrete(self.data[1]) else MarginType.CONTINUOUS

    @computed_field
    @property
    def empirical_copula(self) -> EmpiricalCopula:
        return EmpiricalCopula(data=self.data)

    @staticmethod
    def _get_multipliers(val: float, val_lower: float, val_upper: float) -> tuple[float, float]:
        if np.isclose(val_lower, val_upper):
            return 0.0, 1.0

        length = val_upper - val_lower
        multiplier_lower = (val_upper - val) / length
        multiplier_upper = (val - val_lower) / length
        return multiplier_lower, multiplier_upper

    def __call__(self, u: float, v: float) -> float:
        match (self.first_margin_type, self.second_margin_type):
            case (MarginType.CONTINUOUS, MarginType.CONTINUOUS):
                return self.empirical_copula(np.array([u, v]))
            case (MarginType.CONTINUOUS, MarginType.DISCRETE):
                v_lower, v_upper = step_function(data=self.data[1], arg=v)
                v_lower_multiplier, v_upper_multiplier = self._get_multipliers(v, v_lower, v_upper)

                empirical_copula_lower = self.empirical_copula(np.array([u, v_lower]))
                empirical_copula_upper = self.empirical_copula(np.array([u, v_upper]))

                return v_lower_multiplier * empirical_copula_lower + v_upper_multiplier * empirical_copula_upper

            case (MarginType.DISCRETE, MarginType.CONTINUOUS):
                u_lower, u_upper = step_function(data=self.data[0], arg=u)
                u_lower_multiplier, u_upper_multiplier = self._get_multipliers(u, u_lower, u_upper)

                empirical_copula_lower = self.empirical_copula(np.array([u_lower, v]))
                empirical_copula_upper = self.empirical_copula(np.array([u_upper, v]))

                return u_lower_multiplier * empirical_copula_lower + u_upper_multiplier * empirical_copula_upper

            case (MarginType.DISCRETE, MarginType.DISCRETE):
                u_lower, u_upper = step_function(data=self.data[0], arg=u)
                u_lower_multiplier, u_upper_multiplier = self._get_multipliers(u, u_lower, u_upper)

                v_lower, v_upper = step_function(data=self.data[1], arg=v)
                v_lower_multiplier, v_upper_multiplier = self._get_multipliers(v, v_lower, v_upper)

                empirical_copula_ll = self.empirical_copula(np.array([u_lower, v_lower]))
                empirical_copula_lu = self.empirical_copula(np.array([u_lower, v_upper]))
                empirical_copula_ul = self.empirical_copula(np.array([u_upper, v_lower]))
                empirical_copula_uu = self.empirical_copula(np.array([u_upper, v_lower]))

                return (
                    u_lower_multiplier * v_lower_multiplier * empirical_copula_ll
                    + u_lower_multiplier * v_upper_multiplier * empirical_copula_lu
                    + u_upper_multiplier * v_lower_multiplier * empirical_copula_ul
                    + u_upper_multiplier * v_upper_multiplier * empirical_copula_uu
                )

            case _:
                msg = f"Unknown extension type ({self.first_margin_type}, {self.second_margin_type})"
        raise ValueError(msg)
