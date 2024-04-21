import pathlib

S3_BUCKET = 'cloud-pashentsevw-default'
S3_FOLDER = pathlib.Path('hardml/recsys/lesson9/')

RANDOM_STATE = 94

DATA_PATH = S3_FOLDER
ARTIFACTS_PATH = S3_FOLDER / 'artifacts'
SUBMISSION_PATH = S3_FOLDER / 'submissions'

AT_K_CAND = 50
AT_K_RANK = 20