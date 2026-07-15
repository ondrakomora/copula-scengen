from abc import ABC, abstractmethod

import pandas as pd


class BaseScenarioGenerator(ABC):
    @abstractmethod
    def generate(self, data: pd.DataFrame, n_scenarios: int) -> pd.DataFrame:
        """Generate scenarios based on the provided data."""
