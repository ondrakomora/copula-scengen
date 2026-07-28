import numpy as np

from copula_scengen.copula_sample_generators.copula_sample2d import CopulaSample2D


def test_copula_sample_2d_is_zero_at_rank_zero() -> None:
    sample = CopulaSample2D.initialize(max_rank=4)
    sample.assign(rank=2)

    result = sample(np.arange(5))

    np.testing.assert_array_equal(result, np.array([0.0, 0.0, 0.25, 0.25, 0.25]))
