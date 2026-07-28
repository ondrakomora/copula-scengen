import warnings

import numpy as np
import pandas as pd
import pytest

from copula_scengen.copula.copula_sample import CopulaSample
from copula_scengen.copula_sample_transformers import CopulaSampleTransformer, ExtendedCopulaSampleTransformer
from copula_scengen.functions.discrete_transformation_bounds import discrete_transformation_bounds


@pytest.mark.parametrize("transformer", [CopulaSampleTransformer(), ExtendedCopulaSampleTransformer()])
def test_discrete_transformer_preserves_empirical_counts_at_exact_breakpoints(transformer: object) -> None:
    data = pd.DataFrame({"x": [0, 1, 2, 2]})
    sample = CopulaSample.initialize(max_rank=4)

    result = transformer.transform(data, sample)  # type: ignore[union-attr]

    assert result["x"].tolist() == [0.0, 1.0, 2.0, 2.0]


@pytest.mark.parametrize("transformer", [CopulaSampleTransformer(), ExtendedCopulaSampleTransformer()])
@pytest.mark.parametrize(
    "values",
    [[-1, 0, 1], [2, 2, 2], [0, 2, 2]],
    ids=["negative", "nonzero-minimum", "gapped"],
)
def test_discrete_transformer_rejects_unnormalized_support(transformer: object, values: list[int]) -> None:
    data = pd.DataFrame({"x": values})
    sample = CopulaSample.initialize(max_rank=2)

    with pytest.raises(ValueError, match="contiguous integer values starting at zero"):
        transformer.transform(data, sample)  # type: ignore[union-attr]


def test_extended_discrete_transformer_handles_constant_margin_without_warning() -> None:
    data = pd.DataFrame({"x": [0, 0, 0]})
    sample = CopulaSample.initialize(max_rank=2)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ExtendedCopulaSampleTransformer().transform(data, sample)

    assert result["x"].tolist() == [0.0, 0.0]


def test_extended_inverse_ecdf_is_finite_for_degenerate_cumulative_values() -> None:
    transformer = ExtendedCopulaSampleTransformer()
    cumulative = np.array([0.0, 0.0, 1.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = transformer._extended_inverse_ecdf(cumulative, np.array([0.0, 0.5, 1.0, 1.1]))  # noqa: SLF001

    assert np.isfinite(result).all()


def test_discrete_bounds_assign_exact_cdf_intervals_to_the_matching_support_value() -> None:
    cumulative = np.array([0.25, 0.5, 1.0])
    lower_args = np.arange(4) / 4
    upper_args = np.arange(1, 5) / 4

    lower, upper = discrete_transformation_bounds(cumulative, lower_args, upper_args)

    np.testing.assert_array_equal(lower, np.array([0, 1, 2, 2]))
    np.testing.assert_array_equal(upper, np.array([0, 1, 2, 2]))
