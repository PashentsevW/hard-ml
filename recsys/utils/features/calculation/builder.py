import logging
from typing import Generic, Iterator, Sequence

import pandas

from .calcer import FeatureCalcerBase
from .types import TSample


class FeaturesBuilder(Generic[TSample]):
    def __init__(self, feature_calcers: Iterator[FeatureCalcerBase]) -> None:
        self.feature_calcers = feature_calcers

    def join_features(self,
                      sample_columns: Sequence[str],
                      feature_dfs: Sequence[TSample]) -> TSample:
        raise NotImplementedError

    def calculate_on(self, sample: TSample) -> TSample:
        feature_dfs = []
        for calcer in self.feature_calcers:
            feature_df = calcer.calculate_on(sample)

            logging.info('Got features from "%s" calcer: %s',
                         calcer.calcer_id,
                         feature_df.columns)

            feature_dfs.append(feature_df)

        return self.join_features(sample.columns, feature_dfs)


class PandasFeaturesBuilder(FeaturesBuilder[pandas.DataFrame]):
    def join_features(self,
                      sample_columns: Sequence[str],
                      feature_dfs: Sequence[pandas.DataFrame]) -> pandas.DataFrame:
        features_df = feature_dfs[0]
        for feature_df in feature_dfs[1:]:
            logging.info('Join features: %s', feature_df.columns)

            features_df = features_df.merge(feature_df, how='outer', on=sample_columns)
        
        logging.info('Got features with shape: %s', features_df.shape)
        
        return features_df
