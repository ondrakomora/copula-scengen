import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr


class CopulaSample2D(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_rank: int
    _cache: np.ndarray = PrivateAttr(default=np.array([]))

    @classmethod
    def initialize(cls, max_rank: int) -> "CopulaSample2D":
        obj = cls(max_rank=max_rank)
        obj._cache = np.zeros(max_rank)
        return obj

    def __call__(self, arg: np.ndarray) -> float | np.ndarray:
        arr = np.asarray(arg) - 1
        return self._cache[arr]

    def assign(self, rank: int) -> "CopulaSample2D":
        new_obj = CopulaSample2D(max_rank=self.max_rank)
        new_cache = self._cache.copy()

        new_cache[rank - 1 :] += 1.0 / self.max_rank

        new_obj._cache = new_cache
        return new_obj
