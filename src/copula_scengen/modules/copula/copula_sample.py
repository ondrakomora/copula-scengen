import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class CopulaSample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ranks: np.ndarray
    max_rank: int

    @model_validator(mode="after")
    def validate_ranks(self) -> "CopulaSample":
        """Ensure ranks are integers, within range [1, max_rank], and unique within each column."""
        ranks = self.ranks

        if not np.issubdtype(ranks.dtype, np.integer):
            msg = "Ranks must contain only integers"
            raise ValueError(msg)

        if np.any(ranks < 1) or np.any(ranks > self.max_rank):
            msg = f"Ranks must be integers in the range [1, {self.max_rank}]"
            raise ValueError(msg)

        # Uniqueness check per column
        for j in range(ranks.shape[1]):
            col = ranks[:, j]
            if np.unique(col).size != col.size:
                msg = f"Column {j} of ranks contains duplicate values"
                raise ValueError(msg)

        return self

    @classmethod
    def initialize(cls, max_rank: int) -> "CopulaSample":
        """Initialize ranks with values from 1 to max_rank (cyclically if needed)."""
        ranks = np.arange(1, max_rank + 1).reshape((max_rank, 1))
        return cls(ranks=ranks, max_rank=max_rank)

    def retrieve_scenarios(self, scenario_idxs: list[int]) -> np.ndarray:
        return self.ranks[scenario_idxs, :]

    def extend(self, new_ranks: np.ndarray[int]) -> "CopulaSample":
        extended_ranks = np.append(self.ranks, new_ranks.reshape((self.max_rank, 1)), axis=1)

        return CopulaSample(
            ranks=extended_ranks,
            max_rank=self.max_rank,
        )
