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

        i_arr = np.arange(1, max_rank + 1)

        v_val = rank / max_rank
        tc_args_upper = np.column_stack((i_arr / max_rank, np.full(i_arr.shape, v_val)))
        tc_args_lower = np.column_stack(((i_arr - 1) / max_rank, np.full(i_arr.shape, v_val)))

        cache_matrix = np.zeros((num_margins, max_rank), dtype=float)

        for margin, (copula_sample, target_copula) in enumerate(zip(copula_samples, target_copulas, strict=False)):
            # vectorized evaluations
            cs_eval_1 = copula_sample(i_arr)
            tc_eval_1 = target_copula(args=tc_args_upper)

            # first delta term
            delta = np.sum(np.abs(cs_eval_1 + 1.0 / max_rank - tc_eval_1))

            # second vectorized part
            cs_eval_2 = copula_sample(i_arr - 1)
            tc_eval_2 = target_copula(args=tc_args_lower)

            delta_arr = np.abs(cs_eval_2 - tc_eval_2) - np.abs(cs_eval_2 + 1.0 / max_rank - tc_eval_2)
            cache_matrix[margin] = delta + np.cumsum(delta_arr)

        instance = cls()
        instance._cache_matrix = cache_matrix
        return instance

    def __call__(self, margin: int, rank: int) -> float:
        return self._cache_matrix[margin, rank - 1]
