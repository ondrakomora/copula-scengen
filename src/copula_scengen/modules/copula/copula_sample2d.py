from pydantic import BaseModel, PrivateAttr


class CopulaSample2D(BaseModel):
    max_rank: int

    _cache: dict[int, float] = PrivateAttr(default_factory=dict)

    @classmethod
    def initialize(cls, max_rank: int) -> "CopulaSample2D":
        obj = cls(max_rank=max_rank)
        obj._cache = dict.fromkeys(range(1, max_rank + 1), 0.0)
        return obj

    def evaluate(self, arg: int) -> float:
        return self._cache.get(arg, 0.0)

    def assign(self, rank: int) -> "CopulaSample2D":
        new_obj = CopulaSample2D(max_rank=self.max_rank)
        new_obj._cache = self._cache.copy()
        for idx in range(rank, self.max_rank + 1):
            new_obj._cache[idx] = self._cache.get(idx, 0.0) + 1.0 / self.max_rank
        return new_obj
