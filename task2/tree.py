import numpy as np
from typing import Iterable, Dict


class UpliftTreeRegressor(object):
    def __init__(self,
                 max_depth: int = 3,
                 min_samples_leaf: int = 1000,
                 min_samples_leaf_treated: int = 300,
                 min_samples_leaf_control: int = 300):
        pass

    def fit(self,
            X: np.ndarray, 
            treatment: np.ndarray, 
            y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> Iterable[float]:
        raise NotImplementedError


def _check(model_params: Dict[str, int],
           X: np.ndarray,
           treatment: np.ndarray,
           y: np.ndarray,
           uplift_true: np.ndarray) -> bool:

    assert X.shape[0] == treatment.shape[0]
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == uplift_true.shape[0]

    model = UpliftTreeRegressor(**model_params)
    model.fit(X, treatment, y)

    uplift_pred = model.predict(X)

    assert uplift_pred.shape == uplift_true.shape
    assert not np.any(np.isnan(uplift_pred))

    return np.allclose(uplift_true, uplift_pred, atol=1e-5)


if __name__ == '__main__':
    model_params = {'max_depth': 3,
                    'min_samples_leaf': 6000, 
                    'min_samples_leaf_treated': 2500, 
                    'min_samples_leaf_control': 2500,}
    X = np.load('task2/example_X.npy')
    treatment = np.load('task2/example_treatment.npy')
    y = np.load('task2/example_y.npy')
    uplift_true = np.load('task2/example_preds.npy')

    print(f'X shape: {X.shape}', 
          f'treatment shape: {treatment.shape}',
          f'y shape: {y.shape}',
          f'uplift_true shape: {uplift_true.shape}',)

    print(_check(model_params, X, treatment, y, uplift_true))