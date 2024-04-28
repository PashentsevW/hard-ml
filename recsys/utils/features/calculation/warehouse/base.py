from abc import abstractmethod
from collections.abc import Mapping


class DataWarehouseBase(Mapping):
    @abstractmethod
    def register(self, data_id: str, **kwargs) -> None:
        raise NotImplementedError
