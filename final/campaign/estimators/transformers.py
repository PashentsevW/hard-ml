from typing import List
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted


class LabelEncoder(BaseEstimator, TransformerMixin):
    name = 'label_encoder'

    def __init__(self,
                 columns: List[str],
                 copy: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.columns = {c: None for c in columns}
        self.copy = copy
        self._fitted = False

    def fit(self,
            X: DataFrame,
            y: Series = None,
            **fit_params):
        if y is None:
            check_array(X, dtype=None, force_all_finite='allow-nan')
        else:
            check_X_y(X, y, dtype=None, force_all_finite='allow-nan')

        for column in self.columns:
            values = X.loc[X[column].notna(), column].unique()

            self.columns[column] = {values[i]: i 
                                    for i in range(len(values))}
        
        self._fitted = True
        return self

    def transform(self, 
                  X: DataFrame,
                  y: Series = None) -> DataFrame:
        if y is None:
            check_array(X, dtype=None, force_all_finite='allow-nan')
        else:
            check_X_y(X, y, dtype=None, force_all_finite='allow-nan')
        check_is_fitted(self, '_fitted')

        if self.copy:
            X_transformed = X.copy()
        else:
            X_transformed = X

        for column in self.columns:
            X_transformed.loc[:, column] = X[column].map(self.columns[column])

        return X_transformed