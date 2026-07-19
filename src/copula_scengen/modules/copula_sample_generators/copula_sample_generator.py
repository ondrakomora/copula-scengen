import numpy as np
import pandas as pd

from copula_scengen.modules.copula.base import CopulaProvider
from copula_scengen.modules.copula.copula_sample import CopulaSample
from copula_scengen.modules.copula.copula_sample2d import CopulaSample2D
from copula_scengen.modules.copula.extended_empirical_copula_provider import ExtendedEmpiricalCopulaProvider
from copula_scengen.modules.copula_sample_generators.base import CopulaSampleGenerationStrategy
from copula_scengen.modules.copula_sample_generators.deviation_cache import DeviationCache


class CopulaSampleGenerator(CopulaSampleGenerationStrategy):
    def __init__(self, copula_provider: CopulaProvider | None = None) -> None:
        self._copula_provider = copula_provider or ExtendedEmpiricalCopulaProvider()

    def create(self, data: pd.DataFrame, n_scenarios: int) -> CopulaSample:
        copula_sample = CopulaSample.initialize(max_rank=n_scenarios, n_margins=data.shape[1])
        for new_margin in range(1, data.shape[1]):
            copula_sample = self._assign_ranks_to_margin(
                copula_sample=copula_sample,
                data=data,
                margin=new_margin,
                n_scenarios=n_scenarios,
            )
        return copula_sample

    def _assign_ranks_to_margin(
        self,
        copula_sample: CopulaSample,
        data: pd.DataFrame,
        margin: int,
        n_scenarios: int,
    ) -> CopulaSample:
        available = np.ones(n_scenarios, dtype=bool)

        copula_samples_2d = [CopulaSample2D.initialize(n_scenarios) for _ in range(margin)]
        target_grids = [
            DeviationCache.precompute_target_grid(
                target_copula=self._copula_provider.get(data=data, margins=[prior_margin, margin]),
                max_rank=n_scenarios,
            )
            for prior_margin in range(margin)
        ]

        new_ranks = np.zeros(n_scenarios, dtype=int)

        all_scenarios = copula_sample.retrieve_scenarios(scenario_idxs=np.arange(n_scenarios))

        for new_rank in range(1, n_scenarios + 1):
            cache = DeviationCache.compute_cache(
                copula_samples=copula_samples_2d,
                target_grids=target_grids,
                rank=new_rank,
            )

            idxs = np.where(available)[0]
            scenario_ranks = all_scenarios[idxs, :margin]

            dev = cache(scenario_ranks).sum(axis=1)

            best_pos = np.argmin(dev)
            best_idx = idxs[best_pos]
            best_scenario = all_scenarios[best_idx, :]

            available[best_idx] = False
            new_ranks[best_idx] = new_rank

            for cs2d, rank in zip(copula_samples_2d, best_scenario, strict=False):
                cs2d.assign(rank=rank)

        return copula_sample.extend(new_ranks=new_ranks)
