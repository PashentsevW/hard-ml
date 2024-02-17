import pathlib

S3_BUCKET = 'cloud-pashentsevw-default'
S3_FOLDER = pathlib.Path('hardml/recsys/lesson2/')

RANDOM_STATE = 94

DATA_PATH = S3_FOLDER
SUBMISSION_PATH = S3_FOLDER / 'submissions'

AT_K = 10
