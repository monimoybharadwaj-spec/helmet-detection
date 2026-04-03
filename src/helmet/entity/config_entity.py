import os
from dataclasses import dataclass
from src.helmet.constants.constants import *

# Model Trainer Config
@dataclass
class ModelTrainerConfig:
    data_yaml: str = DATA_YAML_PATH
    model_name: str = MODEL_NAME

    epochs: int = EPOCHS
    imgsz: int = IMAGE_SIZE
    batch: int = BATCH_SIZE

    optimizer: str = OPTIMIZER
    lr0: float = LEARNING_RATE

    cache : bool = True
    workers: int = NUM_WORKERS

    project: str = RUNS_DIR
    name: str = EXPERIMENT_NAME

    device : str = DEVICE

# Model Evaluation Config
@dataclass
class ModelEvaluationConfig:
    conf_threshold: float = CONF_THRESHOLD
    iou_threshold: float = IOU_THRESHOLD
    metric: str = METRIC_TO_TRACK

# Prediction Config
@dataclass
class PredictionConfig:
    model_path: str = "runs/detect/exp/weights/best.pt"
    source_dir: str = TEST_IMAGE_DIR