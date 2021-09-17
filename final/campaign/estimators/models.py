import numpy as np
from sklearn.base import BaseEstimator
from causalml.inference.tree import UpliftRandomForestClassifier


class UpliftRandomForest(UpliftRandomForestClassifier, BaseEstimator):
    def __init__(self, 
                 n_estimators=10,
                 max_features=10,
                 random_state=2019,
                 max_depth=5,
                 min_samples_leaf=100,
                 min_samples_treatment=10,
                 n_reg=10,
                 evaluationFunction=None,
                 control_name='0',
                 n_jobs=-1,
                 **kwargs):
        super().__init__(n_estimators,
                         max_features,
                         random_state,
                         max_depth,
                         min_samples_leaf,
                         min_samples_treatment,
                         n_reg,
                         evaluationFunction,
                         control_name,
                         n_jobs,
                         **kwargs)

    def fit(self, X, y, **fit_params):
        w = np.where(fit_params['w'] == 1, '1', '0')
        return super().fit(X.values, w, y.values)

    def predict(self, X):
        return super().predict(X.values, full_output=False).reshape(-1)
