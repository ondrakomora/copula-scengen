import numpy as np

from copula_scengen.copula.base import Copula
from copula_scengen.copula.copula_sample2d import CopulaSample2D


class DeviationCache:
    def __init__(self, cache_matrix: np.ndarray) -> None:
        self._cache_matrix = cache_matrix

    @staticmethod
    def precompute_target_grid(target_copula: Copula, max_rank: int) -> np.ndarray:
        """
        Evaluate ``target_copula`` on the full grid of points ``(i/max_rank, r/max_rank)``.

        Returns a matrix ``M`` of shape ``(max_rank + 1, max_rank + 1)`` with
        ``M[i, r] == target_copula((i / max_rank, r / max_rank))`` for ``i, r`` in
        ``0..max_rank``. The whole grid is evaluated with a single call, replacing the
        previous per-rank re-evaluation inside the greedy assignment loop.

        Delegates to ``Copula.grid``; copulas with extra structure (e.g. the empirical
        copulas) override it with a faster exact cumulative-histogram implementation.
        """
        return target_copula.grid(max_rank)

    @classmethod
    def compute_cache(
        cls,
        copula_samples: list[CopulaSample2D],
        target_grids: list[np.ndarray],
        rank: int,
    ) -> "DeviationCache":
        max_rank = copula_samples[0].max_rank
        num_margins = len(copula_samples)

        i_arr = np.arange(1, max_rank + 1)

        cache_matrix = np.zeros((num_margins, max_rank), dtype=float)

        for margin, (copula_sample, target_grid) in enumerate(zip(copula_samples, target_grids, strict=False)):
            # target copula values sliced from the precomputed grid (column == rank)
            tc_eval_1 = target_grid[1:, rank]
            tc_eval_2 = target_grid[:max_rank, rank]

            # vectorized evaluations of the evolving 2D copula sample
            cs_eval_1 = copula_sample(i_arr)
            cs_eval_2 = copula_sample(i_arr - 1)

            # first delta term
            delta = np.sum(np.abs(cs_eval_1 + 1.0 / max_rank - tc_eval_1))

            delta_arr = np.abs(cs_eval_2 - tc_eval_2) - np.abs(cs_eval_2 + 1.0 / max_rank - tc_eval_2)
            cache_matrix[margin] = delta + np.cumsum(delta_arr)

        return cls(cache_matrix=cache_matrix)

    def __call__(self, ranks: np.ndarray) -> np.ndarray:
        return np.take_along_axis(self._cache_matrix.T, ranks - 1, axis=0)
