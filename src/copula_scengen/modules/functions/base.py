from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, ConfigDict


class BaseFunction(BaseModel, ABC):
    """Base class for all functions."""

    @abstractmethod
    def evaluate(self) -> any:
        """Evaluate the function at given arguments."""


class EmpiricalFunction(BaseFunction, ABC):
    """Base class for functions which are computed empirically."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: np.ndarray
