import numpy as np
import pandas as pd

from copula_scengen.copula.copula_sample import CopulaSample
from copula_scengen.copula_sample_generators.base import CopulaSampleGenerationStrategy
from copula_scengen.copula_sample_transformers.base import CopulaSampleTransformationStrategy
from copula_scengen.scenario_generators.scenario_generator import ScenarioGenerator


def test_scenario_generator_uses_injected_strategies() -> None:
    data = pd.DataFrame({"x": [0.0, 1.0], "y": [2.0, 3.0]})
    copula_sample = CopulaSample.initialize(max_rank=2)
    expected = pd.DataFrame({"x": [100.0, 200.0], "y": [300.0, 400.0]})

    class StubCreationStrategy(CopulaSampleGenerationStrategy):
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], int]] = []

        def create(self, data: pd.DataFrame, n_scenarios: int) -> CopulaSample:
            self.calls.append((list(data.columns), n_scenarios))
            return copula_sample

    class StubTransformationStrategy(CopulaSampleTransformationStrategy):
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], int]] = []

        def transform(self, data: pd.DataFrame, copula_sample: CopulaSample) -> pd.DataFrame:
            self.calls.append((list(data.columns), copula_sample.max_rank))
            return expected

    creation_strategy = StubCreationStrategy()
    transformation_strategy = StubTransformationStrategy()

    generator = ScenarioGenerator(
        copula_sample_generation_strategy=creation_strategy,
        copula_sample_transformation_strategy=transformation_strategy,
    )

    result = generator.generate(data=data, n_scenarios=2)

    assert result.equals(expected)
    assert creation_strategy.calls == [(["x", "y"], 2)]
    assert transformation_strategy.calls == [(["x", "y"], 2)]


def test_scenario_generator_setters_accept_structural_strategies() -> None:
    data = pd.DataFrame({"x": [0.0, 1.0]})
    copula_sample = CopulaSample.initialize(max_rank=2)
    expected = pd.DataFrame({"x": [100.0, 200.0]})

    class CreationStrategy:
        def create(self, data: pd.DataFrame, n_scenarios: int) -> CopulaSample:
            return copula_sample

    class TransformationStrategy:
        def transform(self, data: pd.DataFrame, copula_sample: CopulaSample) -> pd.DataFrame:
            return expected

    generator = ScenarioGenerator()
    generator.set_copula_sample_generation_strategy(CreationStrategy())
    generator.set_copula_sample_transformation_strategy(TransformationStrategy())

    result = generator.generate(data=data, n_scenarios=2)

    assert result.equals(expected)


def test_scenario_generator_default_strategies_generate_dataframe() -> None:
    data = pd.DataFrame(
        {
            "a": np.array([0.1, 0.5, 0.9], dtype=float),
            "b": np.array([1.0, 2.0, 3.0], dtype=float),
        },
    )

    scenarios = ScenarioGenerator().generate(data=data, n_scenarios=3)

    assert scenarios.shape == (3, 2)
    assert list(scenarios.columns) == ["a", "b"]
