# Copula scenario generation method for stochastic programming

This library contains a Python implementantion of scenario generation method for scenario generation which is based on method introduced in article [Michal Kaut (2014)](https://work.michalkaut.net/papers_etc/sg-cop_heur.pdf) and its extension for discrete data based on diploma thesis [Komora (2024)](https://dspace.cuni.cz/handle/20.500.11956/190636).

## Usage

Simple usage is as follows.

```python
from copula_scengen import ScenarioGenerator

generator = ScenarioGenerator()

scenario_generator = ScenarioGenerator(data=datafr)

scenarios_datafr = scenario_generator.generate(n_scenarios=10)
```

**Note**: ScenarioGenerator is a pydantic class. Therefore you need to specify 'data' parameter explicitly. The type of this parameter is 'pd.DataFrame'.

### Expected data format.

It is expected that each column in input dataframe is a separate margin. There are no restrictions for continuous margins. 

For discrete margins, there are following rules:
1. Discrete margins consist of values from zero to some integer n, while each number between is present.
2. Any categorical values, like 'No' and 'Yes', must be mapped by user to integer values (for example 0 and 1). After obtaining scenarios, it is possible to revert to original values.

Discrete margins are automatically inferred by the following function.
```python
def is_discrete(arr: np.ndarray) -> bool:
    """
    Return True if all values in the numpy array are discrete (integers),
    False if any value is continuous (non-integer).
    NaN values are ignored.
    """
    if not np.issubdtype(arr.dtype, np.number):
        msg = "Array must contain numeric values only."
        raise TypeError(msg)

    # Ignore NaNs, compare integer-casted values to originals
    arr_no_nan = arr[~np.isnan(arr)]
    return np.allclose(arr_no_nan, np.round(arr_no_nan))
```
