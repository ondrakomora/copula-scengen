import numpy as np
import pandas as pd
from loguru import logger

from copula_scengen.modules.generator.copula_sample_generator import CopulaSampleGenerator
from copula_scengen.modules.generator.copula_sample_transformer import CopulaSampleTransformer


def generate_binary_dataset(n_samples: int = 10, n_features: int = 3) -> pd.DataFrame:
    data = np.random.randint(0, 2, size=(n_samples, n_features))
    columns = [f"f{i}" for i in range(n_features)]
    return pd.DataFrame(data, columns=columns)


if __name__ == "__main__":
    data = generate_binary_dataset(n_samples=1000, n_features=20)

    copula_sample = CopulaSampleGenerator(data=data).generate(n_scenarios=10)

    logger.info(copula_sample)

    scenarios = CopulaSampleTransformer(data=data, copula_sample=copula_sample).transform()

    logger.info(scenarios)
