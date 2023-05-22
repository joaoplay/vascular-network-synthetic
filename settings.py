import os

OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./files")
MODELS_DIR_NAME = 'models'
CHECKPOINTS_DIR_NAME = 'checkpoints'
PROCESSED_DATA_DIR_NAME = 'processed_data'

WANDB_PROJECT_NAME = 'vascular-networks'

os.environ["WANDB_DIR"] = OUTPUT_PATH
