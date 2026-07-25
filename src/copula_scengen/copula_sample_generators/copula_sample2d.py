import numpy as np


class CopulaSample2D:
    """Track a two-dimensional empirical copula during rank assignment."""

    def __init__(self, max_rank: int) -> None:
        self.max_rank = max_rank
        self._cache = np.zeros(max_rank)

    @classmethod
    def initialize(cls, max_rank: int) -> "CopulaSample2D":
        """Create an empty two-dimensional sample with the given rank limit."""
        return cls(max_rank=max_rank)

    def __call__(self, arg: np.ndarray) -> float | np.ndarray:
        """Return empirical copula values at the supplied ranks."""
        arr = np.asarray(arg) - 1
        return self._cache[arr]

    def assign(self, rank: int) -> None:
        """Incorporate one assigned rank into the empirical copula values."""
        self._cache[rank - 1 :] += 1.0 / self.max_rank
