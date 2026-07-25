from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class BaseScenarioGenerator(Protocol):
    """Define the interface for generating scenarios from observed data."""

    def generate(self, data: pd.DataFrame, n_scenarios: int) -> pd.DataFrame:
        """Generate scenarios based on the provided data."""
        ...
