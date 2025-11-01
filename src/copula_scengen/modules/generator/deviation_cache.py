from math import fabs

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from copula_scengen.modules.copula.copula_sample2d import CopulaSample2D
from copula_scengen.modules.copula.empirical_copula import EmpiricalCopula


class DeviationCache(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _cache_matrix: np.ndarray = PrivateAttr(default=np.array([]))

    @classmethod
    def compute_cache(
        cls,
        copula_samples: list[CopulaSample2D],
        target_copulas: list[EmpiricalCopula],
        rank: int,
    ) -> "DeviationCache":
        max_rank = copula_samples[0].max_rank

        num_margins = len(copula_samples)
        cache_matrix = np.zeros((num_margins, max_rank), dtype=float)

        for margin, copula_sample in enumerate(copula_samples):
            target_copula = target_copulas[margin]
            delta = sum(
                fabs(
                    copula_sample(arg=i)
                    + 1.0 / max_rank
                    - target_copula(args=np.array([i / max_rank, rank / max_rank]))
                )
                for i in range(1, max_rank + 1)
            )

            for i in range(1, max_rank + 1):
                copula_sample_eval = copula_sample(i - 1)
                target_copula_eval = target_copula(args=np.array([(i - 1) / max_rank, rank / max_rank]))
                delta += fabs(copula_sample_eval - target_copula_eval) - fabs(
                    copula_sample_eval + 1.0 / max_rank - target_copula_eval
                )
                cache_matrix[margin, i - 1] = delta

        instance = cls()
        instance._cache_matrix = cache_matrix
        return instance

    def evaluate(self, margin: int, rank: int) -> float:
        return self._cache_matrix[margin, rank - 1]
