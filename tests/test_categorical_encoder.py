import pandas as pd

from copula_scengen.preprocessing.categorical_encoder import CategoricalEncoder


def test_encoder_normalizes_and_restores_discrete_numeric_support() -> None:
    data = pd.DataFrame({"x": [-2, 0, 3, 3]})
    encoder = CategoricalEncoder()

    encoded, mapping = encoder.encode(data)
    decoded = encoder.decode(encoded, mapping)

    assert encoded["x"].tolist() == [0, 1, 2, 2]
    assert decoded.equals(data)
