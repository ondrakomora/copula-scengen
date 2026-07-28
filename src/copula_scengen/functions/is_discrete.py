import numpy as np


def is_discrete(arr: np.ndarray) -> bool:
    """
    Return True if all values in the numpy array are discrete (integers),
    False if any value is continuous (non-integer).
    NaN values are ignored.
    """
    if not np.issubdtype(arr.dtype, np.number):
        msg = "Array must contain numeric values only."
        raise TypeError(msg)

    # Ignore NaNs, then compare finite values to their rounded forms.
    arr_no_nan = arr[~np.isnan(arr)]
    if not np.isfinite(arr_no_nan).all():
        return False
    return bool(np.equal(arr_no_nan, np.round(arr_no_nan)).all())
