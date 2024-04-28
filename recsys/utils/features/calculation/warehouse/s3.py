import logging
from pathlib import Path
from typing import Callable, Iterator, Union

import pandas
import polars

from .base import DataWarehouseBase
from ....io.s3 import DataFrameType, download_dataframe

OutputDataType = Union[pandas.DataFrame, polars.DataFrame]

def _pass_through(dataframe: OutputDataType) -> OutputDataType:
    return dataframe


class S3DataWarehouse(DataWarehouseBase):
    def __init__(self, s3_client) -> None:
        super().__init__()
        self.client = s3_client
        self.metadata = {}
        self.cache = {}

    def register(self,
                 data_id: str,
                 s3_bucket: str,
                 s3_path: Path,
                 preprocessing: Callable[[OutputDataType], OutputDataType] = _pass_through,
                 output_format: DataFrameType = DataFrameType.PANDAS,
                 lazy_loading: bool = True) -> None:
        self.metadata[data_id] = (s3_bucket, s3_path, preprocessing, output_format)

        logging.info('Register metadata for "%s" data', data_id)

        if not lazy_loading:
            logging.info('Download "%s" data to cache', data_id)

            self.cache[data_id] = preprocessing(download_dataframe(self.client, s3_bucket, s3_path, output_format))

    def __getitem__(self, key: str) -> OutputDataType:
        if not key in self.cache:
            logging.info('Download "%s" data to cache', key)

            s3_bucket, s3_path, preprocessing, output_format = self.metadata[key]

            self.cache[key] = preprocessing(download_dataframe(self.client, s3_bucket, s3_path, output_format))
        
        return self.cache[key]

    def __iter__(self) -> Iterator:
        return iter(self.metadata)

    def __len__(self) -> int:
        return len(self.metadata)
