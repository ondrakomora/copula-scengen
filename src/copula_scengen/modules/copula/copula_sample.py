import numpy as np


class CopulaSample:
    def __init__(self, ranks: np.ndarray, max_rank: int) -> None:
        self.ranks = ranks
        self.max_rank = max_rank
        self._buffer: np.ndarray | None = None
        self._filled = 0

    @classmethod
    def initialize(cls, max_rank: int, n_margins: int = 1) -> "CopulaSample":
        """Initialize ranks, preallocating capacity for `n_margins` columns for later `extend` calls."""
        buffer = np.zeros((max_rank, max(n_margins, 1)), dtype=int)
        buffer[:, 0] = np.arange(1, max_rank + 1)

        instance = cls(ranks=buffer[:, :1], max_rank=max_rank)
        instance._buffer = buffer
        instance._filled = 1
        return instance

    def retrieve_scenarios(self, scenario_idxs: list[int]) -> np.ndarray:
        return self.ranks[scenario_idxs, :]

    def extend(self, new_ranks: np.ndarray) -> "CopulaSample":
        if self._buffer is not None and self._filled < self._buffer.shape[1]:
            buffer = self._buffer
            filled = self._filled + 1
            buffer[:, self._filled] = new_ranks
            extended_ranks = buffer[:, :filled]
        else:
            buffer = None
            filled = 0
            extended_ranks = np.append(self.ranks, new_ranks.reshape((self.max_rank, 1)), axis=1)

        instance = CopulaSample(ranks=extended_ranks, max_rank=self.max_rank)
        instance._buffer = buffer
        instance._filled = filled
        return instance
