from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class BaseScenarioGenerator(Protocol):
    def generate(self, data: pd.DataFrame, n_scenarios: int) -> pd.DataFrame:
        """Generate scenarios based on the provided data."""
        ...
