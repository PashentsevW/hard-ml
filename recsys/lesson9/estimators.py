from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Tuple

import numpy
import pandas
from sklearn.model_selection import PredefinedSplit

import columns


class SplitterBase(ABC):
    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        pass

    @abstractmethod
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[numpy.ndarray, numpy.ndarray]]:
        pass


class SearcherBase(ABC):
    @abstractmethod
    def fit(self, X=None, y=None, groups=None):
        pass


class LastNSampleSplitter(SplitterBase):
    def __init__(self, n_last: int) -> None:
        super().__init__()
        self.n_last = n_last

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return 1

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[numpy.ndarray, numpy.ndarray]]:
        dataset_df: pandas.DataFrame = X
        dataset_df = dataset_df.sort_values([columns.USER_ID_COLUMN, columns.DT_COLUMN], ascending=[True, True])

        items_ind_by_user = dataset_df.groupby(columns.USER_ID_COLUMN).cumcount(ascending=False)
        
        test_mask = items_ind_by_user < self.n_last

        test_fold = numpy.full(len(dataset_df), -1, dtype=numpy.int_)
        test_fold[test_mask] = 0

        return PredefinedSplit(test_fold).split(X)
