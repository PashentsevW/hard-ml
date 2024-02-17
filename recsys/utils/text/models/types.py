from typing import Iterator, Mapping

import numpy
from gensim.models import KeyedVectors


class PretrainedWordModel(Mapping[str, numpy.ndarray]):
    def __init__(self) -> None:
        super().__init__()
        self.__vector_size = None

    def __getitem__(self, __key: str) -> numpy.ndarray:
        raise NotImplementedError

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def vector_size(self) -> int:
        return self.__vector_size
 
    @vector_size.setter
    def vector_size(self, vector_size: int):
        self.__vector_size = vector_size


class GensimPretraineWordModel(PretrainedWordModel):
    def __init__(self, word_vectors: KeyedVectors) -> None:
        super().__init__()
        self.word_vectors = word_vectors
    
    def __getitem__(self, __key: str) -> numpy.ndarray:
        return self.word_vectors[__key]

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
