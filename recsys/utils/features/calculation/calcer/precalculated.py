from .base import FeatureCalcerBase
from ..types import TSample


class PrecalculatedFeatureCalcer(FeatureCalcerBase[TSample]):
    def get_data_on(self, sample: TSample) -> TSample:
        raise NotImplementedError

    def calculate_on(self, sample: TSample) -> TSample:
        return self.get_data_on(sample)
