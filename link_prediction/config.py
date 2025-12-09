import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

ENTITY_JSON = os.getenv('ENTITY_JSON')
EMB_H5 = os.getenv('EMB_H5')
MODEL_H5 = os.getenv('MODEL_H5')
ACCESS_KEY = os.getenv('ACCESS_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT')
DIM = 400
TOPK = 100
COMPARATOR_TYPE = "cos"
CACHE_DIR = Path("./data")
