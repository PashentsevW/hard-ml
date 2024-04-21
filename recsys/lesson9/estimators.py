from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Tuple

import numpy


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
