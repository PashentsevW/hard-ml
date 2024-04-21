import io
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Union

import botocore
import pandas
import pickle
import polars


class DataFrameType(Enum):
    PANDAS = 0
    POLARS = 1


def list_objects(client: 'botocore.client.S3', busket: str, folder: Optional[Path] = None) -> List[str]:
    return [object['Key'] 
            for object in client.list_objects(Bucket=busket, Prefix=str(folder or ''))['Contents']]


def upload_dataframe(dataframe: pandas.DataFrame,
                     client: 'botocore.client.S3',
                     busket: str,
                     folder: Path) -> None:
    buffer = io.BytesIO()
    dataframe.to_parquet(buffer, index=False)

    result = client.put_object(Body=buffer.getvalue(), Bucket=busket, Key=str(folder))
    if result['ResponseMetadata']['HTTPStatusCode'] != 200:
        raise IOError('File upload to S3 failed!')


def upload_object(obj: Any,
                  client: 'botocore.client.S3',
                  busket: str,
                  folder: Path) -> None:
    buffer = io.BytesIO(pickle.dumps(obj))

    result = client.put_object(Body=buffer.getvalue(), Bucket=busket, Key=str(folder))
    if result['ResponseMetadata']['HTTPStatusCode'] != 200:
        raise IOError('File upload to S3 failed!')


def download_dataframe(
        client: 'botocore.client.S3',
        busket: str,
        filepath: Path,
        dataframe_type: DataFrameType = DataFrameType.PANDAS
    ) -> Union[pandas.DataFrame, polars.DataFrame]:
    obj = client.get_object(Bucket=busket, Key=str(filepath))
    buffer = io.BytesIO(obj['Body'].read())
    
    if dataframe_type == DataFrameType.PANDAS:
        return pandas.read_parquet(buffer)
    elif dataframe_type == DataFrameType.POLARS:
        return polars.read_parquet(buffer)
    else:
        raise NotImplementedError(dataframe_type.name)


def download_object(client: 'botocore.client.S3',
                    busket: str,
                    filepath: Path,) -> Any:
    obj = client.get_object(Bucket=busket, Key=str(filepath))
    buffer = io.BytesIO(obj['Body'].read())

    return pickle.loads(buffer)
