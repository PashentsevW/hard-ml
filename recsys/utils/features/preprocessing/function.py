from typing import Any, Callable

import numpy
from sklearn.utils.validation import check_array, column_or_1d

def function_1d_wrapper(X: numpy.ndarray,
                        func1d: Callable[[Any,], Any],
                        dtype: numpy.dtype) -> numpy.ndarray:
    X = column_or_1d(check_array(X, dtype=None, ensure_2d=False))
    return numpy.array([func1d(Xi) for Xi in X], dtype=dtype)
