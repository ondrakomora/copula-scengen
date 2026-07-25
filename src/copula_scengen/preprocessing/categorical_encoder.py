import numpy as np
import pandas as pd

from copula_scengen.preprocessing.base import DataEncoder

CATEGORICAL_DTYPE_KINDS = ("category", "object")


class CategoricalEncoder(DataEncoder):
    """Encode categorical DataFrame columns as integer codes and decode them."""

    def encode(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """
        Validate `data`, then map categorical columns to integer codes starting at 0.

        Raises a ValueError if any column contains missing or infinite values.
        Returns the encoded DataFrame along with a mapping of column name to the
        sorted array of original category values, to be used with `decode`.
        """
        self._validate(data)

        encoded = data.copy()
        mapping: dict[str, np.ndarray] = {}

        for column in data.columns:
            if self._is_categorical(data[column]):
                categories = np.sort(data[column].unique())
                codes = np.searchsorted(categories, data[column].to_numpy())

                encoded[column] = codes
                mapping[column] = categories

        return encoded, mapping

    def decode(self, data: pd.DataFrame, mapping: dict[str, np.ndarray]) -> pd.DataFrame:
        """Map integer-coded categorical columns in `data` back to their original values."""
        decoded = data.copy()

        for column, categories in mapping.items():
            codes = np.rint(decoded[column].to_numpy()).astype(int)
            decoded[column] = categories[codes]

        return decoded

    def _is_categorical(self, column: pd.Series) -> bool:
        return isinstance(column.dtype, pd.CategoricalDtype) or column.dtype == object

    def _validate(self, data: pd.DataFrame) -> None:
        if data.isna().any().any():
            msg = "Data contains missing values"
            raise ValueError(msg)

        numeric_columns = data.select_dtypes(include=[np.number])
        if not numeric_columns.empty and not np.isfinite(numeric_columns.to_numpy()).all():
            msg = "Data contains infinite values"
            raise ValueError(msg)
