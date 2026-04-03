import os

# Project paths

ARTIFACTS_DIR : str = "artifacts"
DATA_DIR : str = "data"
DATASET_NAME : str = "helmet_detection"
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)
DATA_YAML_PATH = os.path.join(DATASET_PATH, "data.yaml")
RUNS_DIR = "runs"
EXPERIMENT_NAME : str = "exp"

# Device
DEVICE : str = "auto"

# Model Training Constants

MODEL_NAME = "yolov8s.pt"
EPOCHS : int = 10
IMAGE_SIZE : int = 512
BATCH_SIZE : int = 4
CACHE : bool = True
LEARNING_RATE : float = 1e-3
OPTIMIZER : str = "AdamW"
NUM_WORKERS : int = 2

# Model Evaluation

CONF_THRESHOLD : float = 0.25
IOU_THRESHOLD : float = 0.5
SAVE_RESULTS : bool = True
METRIC_TO_TRACK : str = "mAP50"

# Dataset Info

NUM_CLASSES : int = 2
CLASS_NAMES = ["helmet", "no-helmet"]

# Model Prediction

PREDICTION_DIR : str = "prediction"
PRED_CONF_THRESHOLD : float = 0.6
TEST_IMAGE_DIR = os.path.join(DATASET_PATH, "test", "images")