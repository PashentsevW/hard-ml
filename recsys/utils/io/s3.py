import io
from pathlib import Path
from typing import List, Optional

import botocore
import pandas


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



def download_dataframe(client: 'botocore.client.S3', busket: str, filepath: Path) -> pandas.DataFrame:
    obj = client.get_object(Bucket=busket, Key=str(filepath))
    return pandas.read_parquet(io.BytesIO(obj['Body'].read()))
