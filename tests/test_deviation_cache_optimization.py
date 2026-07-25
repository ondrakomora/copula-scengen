"""
Old-vs-new parity tests for the optimized copula-sample generation.

The optimization precomputes each target copula on the full ``(i/m, r/m)`` grid with a
single ``__call__`` (see :meth:`DeviationCache.precompute_target_grid`) instead of
re-evaluating the copula on every rank iteration of the greedy assignment loop.

These tests embed a *reference* implementation that mirrors the original per-rank
evaluation and assert that the refactored code produces bit-identical results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from copula_scengen.copula.copula_sample import CopulaSample
from copula_scengen.copula.copula_sample2d import CopulaSample2D
from copula_scengen.copula.empirical_copula_provider import EmpiricalCopulaProvider
from copula_scengen.copula.extended_empirical_copula_provider import ExtendedEmpiricalCopulaProvider
from copula_scengen.copula_sample_generators.copula_sample_generator import CopulaSampleGenerator
from copula_scengen.copula_sample_generators.deviation_cache import DeviationCache

if TYPE_CHECKING:
    from copula_scengen.copula.base import Copula, CopulaProvider


# --------------------------------------------------------------------------- #
# Reference (pre-optimization) implementation
# --------------------------------------------------------------------------- #
def _reference_compute_cache(
    copula_samples: list[CopulaSample2D],
    target_copulas: list[Copula],
    rank: int,
) -> np.ndarray:
    """Original per-rank cache computation, evaluating the copula directly each call."""
    max_rank = copula_samples[0].max_rank
    num_margins = len(copula_samples)

    i_arr = np.arange(1, max_rank + 1)
    v_val = rank / max_rank
    tc_args_upper = np.column_stack((i_arr / max_rank, np.full(i_arr.shape, v_val)))
    tc_args_lower = np.column_stack(((i_arr - 1) / max_rank, np.full(i_arr.shape, v_val)))

    cache_matrix = np.zeros((num_margins, max_rank), dtype=float)
    for margin, (copula_sample, target_copula) in enumerate(zip(copula_samples, target_copulas, strict=False)):
        cs_eval_1 = copula_sample(i_arr)
        tc_eval_1 = target_copula(args=tc_args_upper)
        delta = np.sum(np.abs(cs_eval_1 + 1.0 / max_rank - tc_eval_1))

        cs_eval_2 = copula_sample(i_arr - 1)
        tc_eval_2 = target_copula(args=tc_args_lower)
        delta_arr = np.abs(cs_eval_2 - tc_eval_2) - np.abs(cs_eval_2 + 1.0 / max_rank - tc_eval_2)
        cache_matrix[margin] = delta + np.cumsum(delta_arr)
    return cache_matrix


def _reference_create(provider: CopulaProvider, data: pd.DataFrame, n_scenarios: int) -> np.ndarray:
    """Original greedy generator producing the copula-sample rank matrix."""
    copula_sample = CopulaSample.initialize(max_rank=n_scenarios, n_margins=data.shape[1])
    for margin in range(1, data.shape[1]):
        available = np.ones(n_scenarios, dtype=bool)
        copula_samples_2d = [CopulaSample2D.initialize(n_scenarios) for _ in range(margin)]
        target_copulas = [provider.get(data=data, margins=[prior, margin]) for prior in range(margin)]
        new_ranks = np.zeros(n_scenarios, dtype=int)
        all_scenarios = copula_sample.retrieve_scenarios(scenario_idxs=np.arange(n_scenarios))

        for new_rank in range(1, n_scenarios + 1):
            cache = _reference_compute_cache(copula_samples_2d, target_copulas, new_rank)
            idxs = np.where(available)[0]
            scenario_ranks = all_scenarios[idxs, :margin]
            dev = np.take_along_axis(cache.T, scenario_ranks - 1, axis=0).sum(axis=1)
            best_pos = np.argmin(dev)
            best_idx = idxs[best_pos]
            best_scenario = all_scenarios[best_idx, :]
            available[best_idx] = False
            new_ranks[best_idx] = new_rank
            for cs2d, rank in zip(copula_samples_2d, best_scenario, strict=False):
                cs2d.assign(rank=rank)
        copula_sample = copula_sample.extend(new_ranks=new_ranks)
    return copula_sample.ranks


# --------------------------------------------------------------------------- #
# Fixtures / datasets
# --------------------------------------------------------------------------- #
def _datasets() -> list[tuple[str, pd.DataFrame, int]]:
    rng = np.random.default_rng(1234)
    return [
        ("continuous_3d", pd.DataFrame(rng.normal(size=(60, 3)), columns=list("abc")), 12),
        ("continuous_2d", pd.DataFrame(rng.normal(size=(25, 2)), columns=["p", "q"]), 8),
        (
            "mixed_discrete",
            pd.DataFrame(
                {
                    "x": rng.normal(size=45),
                    "k": rng.integers(0, 4, size=45).astype(float),
                    "y": rng.normal(size=45),
                },
            ),
            10,
        ),
    ]


_PROVIDERS = [
    ("empirical", EmpiricalCopulaProvider()),
    ("extended", ExtendedEmpiricalCopulaProvider()),
]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("_provider_name", "provider"), _PROVIDERS)
@pytest.mark.parametrize(("_tag", "data", "n_scenarios"), _datasets())
def test_precompute_target_grid_matches_direct_call(
    _provider_name: str, provider: CopulaProvider, _tag: str, data: pd.DataFrame, n_scenarios: int
) -> None:
    copula = provider.get(data=data, margins=[0, 1])
    grid = DeviationCache.precompute_target_grid(target_copula=copula, max_rank=n_scenarios)

    coords = np.arange(n_scenarios + 1) / n_scenarios
    first, second = np.meshgrid(coords, coords, indexing="ij")
    expected = copula(np.column_stack((first.ravel(), second.ravel()))).reshape(n_scenarios + 1, n_scenarios + 1)

    np.testing.assert_array_equal(grid, expected)


@pytest.mark.parametrize(("_provider_name", "provider"), _PROVIDERS)
@pytest.mark.parametrize(("_tag", "data", "n_scenarios"), _datasets())
def test_compute_cache_matches_reference(
    _provider_name: str, provider: CopulaProvider, _tag: str, data: pd.DataFrame, n_scenarios: int
) -> None:
    margin = 2 if data.shape[1] > 2 else 1
    copula_samples = [CopulaSample2D.initialize(n_scenarios) for _ in range(margin)]
    target_copulas = [provider.get(data=data, margins=[prior, margin]) for prior in range(margin)]
    target_grids = [DeviationCache.precompute_target_grid(tc, n_scenarios) for tc in target_copulas]

    # exercise several ranks, mutating the 2D samples between iterations like the real loop
    for rank in range(1, n_scenarios + 1):
        new = DeviationCache.compute_cache(copula_samples=copula_samples, target_grids=target_grids, rank=rank)
        ref = _reference_compute_cache(copula_samples, target_copulas, rank)
        np.testing.assert_allclose(new._cache_matrix, ref)  # noqa: SLF001
        for cs2d in copula_samples:
            cs2d.assign(rank=rank)


@pytest.mark.parametrize(("_provider_name", "provider"), _PROVIDERS)
@pytest.mark.parametrize(("_tag", "data", "n_scenarios"), _datasets())
def test_generator_matches_reference(
    _provider_name: str, provider: CopulaProvider, _tag: str, data: pd.DataFrame, n_scenarios: int
) -> None:
    optimized = CopulaSampleGenerator(copula_provider=provider).create(data=data, n_scenarios=n_scenarios).ranks
    reference = _reference_create(provider=provider, data=data, n_scenarios=n_scenarios)
    np.testing.assert_array_equal(optimized, reference)
