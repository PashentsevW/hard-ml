from typing import Any, Sequence, Union

import numpy
from sklearn.utils.validation import check_array, check_consistent_length, column_or_1d

ItemIdType = Union[int, str]


def _intersection(y_true: Sequence[ItemIdType], y_pred: Sequence[ItemIdType], k: int) -> int:
    if len(y_true) == 0:
        return 0

    y_true = column_or_1d(check_array(y_true, dtype=None, ensure_2d=False))
    y_pred = column_or_1d(check_array(y_pred, dtype=None, ensure_2d=False))

    return numpy.isin(y_pred[:k], y_true, assume_unique=True).any().astype(numpy.int8)


def hitrate_score(y_true: numpy.ndarray, y_pred: numpy.ndarray, k: int) -> float:
    y_true = check_array(y_true, dtype=numpy.object_, ensure_2d=False)
    y_pred = check_array(y_pred, dtype=numpy.object_, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    scores = []
    for i in range(y_true.shape[0]):
        scores.append(_intersection(y_true[i], y_pred[i], k))

    return numpy.array(scores).mean()
