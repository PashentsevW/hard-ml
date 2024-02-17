import logging
from typing import Iterator, Mapping

from .loader import ModelLoader
from .types import PretrainedWordModel


class PretrainedModelRegistry(Mapping[str, PretrainedWordModel]):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(PretrainedModelRegistry, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        super().__init__()
        self.models = {}

    def __getitem__(self, __key: str) -> PretrainedWordModel:
        return self.models[__key]

    def __iter__(self) -> Iterator[str]:
        return self.models

    def __len__(self) -> int:
        return len(self.models)

    def register(self, model_id: str, model_loader: ModelLoader, vector_size: int, **kwargs) -> None:
        model = model_loader.load(model_id, **kwargs)
        model.vector_size = vector_size

        self.models[model_id] = model

        logging.info('Register model "%s", with vectors size %d', model_id, model.vector_size)
