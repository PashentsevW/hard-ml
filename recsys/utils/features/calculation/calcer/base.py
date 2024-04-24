from abc import ABC, abstractmethod
from typing import Generic

from ..types import TSample
from ..wareshouse import DataWarehouseBase


class FeatureCalcerBase(ABC, Generic[TSample]):
    def __init__(self, calcer_id: str, dwh: DataWarehouseBase) -> None:
        self.calcer_id = calcer_id
        self.dwh = dwh

    @abstractmethod
    def get_data_on(self, sample: TSample) -> TSample:
        pass

    @abstractmethod
    def calculate_on(self, sample: TSample) -> TSample:
        pass