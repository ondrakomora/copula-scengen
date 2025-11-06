import time

import numpy as np
import pandas as pd
from loguru import logger

from copula_scengen.modules.generator.copula_sample_generator import CopulaSampleGenerator
from copula_scengen.modules.generator.copula_sample_transformer import CopulaSampleTransformer

random_generator = np.random.default_rng(42)


def generate_binary_dataset(n_samples: int = 10, n_features: int = 3) -> pd.DataFrame:
    data = random_generator.integers(0, 2, size=(n_samples, n_features))
    columns = [f"f{i}" for i in range(n_features)]
    return pd.DataFrame(data, columns=columns)


if __name__ == "__main__":
    data = generate_binary_dataset(n_samples=1000, n_features=100)

    start = time.perf_counter()
    copula_sample = CopulaSampleGenerator(data=data).generate(n_scenarios=20)
    logger.info(f"Copula sample generation took {time.perf_counter() - start:.3f}s")

    start = time.perf_counter()
    scenarios = CopulaSampleTransformer(data=data).transform(copula_sample=copula_sample)
    logger.info(f"Scenario transformation took {time.perf_counter() - start:.3f}s")

    logger.debug(copula_sample)
    logger.debug(scenarios)
