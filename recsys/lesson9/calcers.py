import logging

import pandas

from utils.features.calculation.calcer import FeatureCalcerBase
from utils.features.calculation.warehouse import DataWarehouseBase

import columns


class UserItemMatrixCalcer(FeatureCalcerBase[pandas.DataFrame]):
    def __init__(self, calcer_id: str, dwh: DataWarehouseBase) -> None:
        super().__init__(calcer_id, dwh)

    def get_data_on(self, sample: pandas.DataFrame) -> pandas.DataFrame:
        train_df: pandas.DataFrame = self.dwh['train']

        logging.info('Got train data with shape: %s', train_df.shape)

        return sample.merge(train_df,
                            how='left',
                            on=[columns.USER_ID_COLUMN, columns.ITEM_ID_COLUMN, columns.DT_COLUMN])

    def calculate_on(self, sample: pandas.DataFrame) -> pandas.DataFrame:
        user_item_on_date_df = self.get_data_on(sample)

        user_item_df = (user_item_on_date_df
                        .groupby([columns.USER_ID_COLUMN, columns.ITEM_ID_COLUMN],
                                 as_index=False)
                        .agg({columns.TARGET_COLUMN: lambda x: (x == 1).any()}))

        logging.info('Got user item data with shape %s', user_item_df.shape)

        return user_item_df
