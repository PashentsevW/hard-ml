import numpy
from sklearn.metrics import make_scorer


def _hitrate_score(y_true: numpy.ndarray, y_pred: numpy.ndarray, k: int) -> float:
    values = []
    for i in range(y_true.shape[0]):
        if y_true[i].shape[0] == 0:
            continue
        values.append(int(numpy.intersect1d(y_true[i], y_pred[i][:k], assume_unique=True).shape[0] > 0))
    return numpy.asarray(values).mean()


hitrate_at10 = make_scorer(_hitrate_score, k=10)
