def uniform_distribution_function(x: float) -> float:
    """CDF of standard uniform distribution."""
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x
