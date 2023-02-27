import os
from dotenv import load_dotenv

from src.paths import PARENT_DIR

# load key-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR / '.env')

FS_PROJECT_NAME = 'taxi_demand_vls'
try:
    FS_API_KEY = os.environ['FS_API_KEY']
except:
    raise Exception('Create an .env file on the project root with the FS_API_KEY')

FS_GROUP_NAME = 'time_series_hourly_feature_group'
FS_GROUP_VERSION = 1
FS_VIEW_NAME = 'time_series_hourly_feature_view'
FS_VIEW_VERSION = 1
MODEL_NAME = "taxi_demand_predictor_next_hour"
MODEL_VERSION = 1

N_FEATURES = 24 * 28