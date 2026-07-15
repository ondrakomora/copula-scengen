import numpy as np


class CopulaSample2D:
    def __init__(self, max_rank: int) -> None:
        self.max_rank = max_rank
        self._cache = np.zeros(max_rank)

    @classmethod
    def initialize(cls, max_rank: int) -> "CopulaSample2D":
        return cls(max_rank=max_rank)

    def __call__(self, arg: np.ndarray) -> float | np.ndarray:
        arr = np.asarray(arg) - 1
        return self._cache[arr]

    def assign(self, rank: int) -> None:
        self._cache[rank - 1 :] += 1.0 / self.max_rank
