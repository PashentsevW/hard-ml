from typing import Sequence, Union

import numpy
from sklearn.utils.validation import check_array, check_consistent_length, column_or_1d

ItemIdType = Union[int, str]


def _intersection(y_true: Sequence[ItemIdType], y_pred: Sequence[ItemIdType], k: int) -> int:
    if len(y_true) == 0:
        return 0

    y_true = column_or_1d(check_array(y_true, dtype=None, ensure_2d=False))
    y_pred = column_or_1d(check_array(y_pred, dtype=None, ensure_2d=False))

    return numpy.isin(y_pred[:k], y_true, assume_unique=True).any().astype(numpy.int8)


def _ndcg(y_true: Sequence[ItemIdType], y_pred: Sequence[ItemIdType], k: int) -> float:
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0

    y_true = column_or_1d(check_array(y_true, dtype=None, ensure_2d=False))
    y_pred = column_or_1d(check_array(y_pred, dtype=None, ensure_2d=False))

    dcg = numpy.where(numpy.isin(y_pred[:k], y_true),
                      1 / numpy.log2(1 + numpy.arange(1, len(y_pred[:k]) + 1)),
                      0)
    dcg = dcg.sum()

    idcg = (1 / numpy.log2(1 + numpy.arange(1, min(len(y_pred[:k]), len(y_true)) + 1))).sum()

    return dcg / idcg if idcg else 0


def hitrate_score(y_true: numpy.ndarray, y_pred: numpy.ndarray, k: int) -> float:
    y_true = check_array(y_true, dtype=numpy.object_, ensure_2d=False)
    y_pred = check_array(y_pred, dtype=numpy.object_, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    scores = []
    for i in range(y_true.shape[0]):
        scores.append(_intersection(y_true[i], y_pred[i], k))

    return numpy.array(scores).mean()


def ndcg_score(y_true: numpy.ndarray, y_pred: numpy.ndarray, k: int) -> float:
    y_true = check_array(y_true, dtype=numpy.object_, ensure_2d=False)
    y_pred = check_array(y_pred, dtype=numpy.object_, force_all_finite='allow-nan', ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    scores = []
    for i in range(y_true.shape[0]):
        scores.append(_ndcg(y_true[i], y_pred[i], k))

    return numpy.array(scores).mean()
