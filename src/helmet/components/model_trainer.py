import os
import sys
from ultralytics import YOLO
from src.helmet.entity.artifact_entity import ModelTrainerArtifact
from src.helmet.entity.config_entity import ModelTrainerConfig
from src.helmet.logger.logger import logging
from src.helmet.exception.exception import CustomException
from src.helmet.utils.device import choose_device

class ModelTrainer:
    
    def __init__(self, model_trainer_config : ModelTrainerConfig):
        try:
            self.model_trainer_config = model_trainer_config
            logging.info("Model Trainer Initialized.")
        except Exception as e:
            raise CustomException(e, sys)
        
    def model_training(self):
        try:
            logging.info("Starting model training...")

            # Loading the model
            model = YOLO(self.model_trainer_config.model_name)
            logging.info(f"YOLOV8 model : {self.model_trainer_config.model_name} loaded successfully.")

            # Setting up device
            device = choose_device(self.model_trainer_config.device)
            print(f"Using {device}.")
            logging.info(f"Using {device} device.")

            # Model training
            results = model.train(
                data = self.model_trainer_config.data_yaml,
                epochs = self.model_trainer_config.epochs,
                imgsz = self.model_trainer_config.imgsz,
                batch = self.model_trainer_config.batch,
                optimizer = self.model_trainer_config.optimizer,
                workers = self.model_trainer_config.workers,
                lr0 = self.model_trainer_config.lr0,
                cache = self.model_trainer_config.cache,
                amp = True,
                device = device,
                project = self.model_trainer_config.project,
                name = self.model_trainer_config.name,
                exist_ok = True
            )
            logging.info("Model training successful.")

            # Best model path
            model_path = os.path.join(self.model_trainer_config.project, self.model_trainer_config.name, "weights", "best.pt")
            logging.info(f"Best model saved at {model_path}.")

            # Evaluation
            metrics = model.val()
            logging.info("Model evaluation completed.")

            # Extracting key metrics
            metrics_dict = {
                "mAP50" : float(metrics.box.map50),
                "mAP50-95" : float(metrics.box.map),
                "precision" : float(metrics.box.mp),
                "recall" : float(metrics.box.mr)
            }
            logging.info(f"Evaluation metrics : {metrics_dict}")

            # Returning artifacts
            return ModelTrainerArtifact(model_path = model_path, metrics = metrics_dict)
        
        except Exception as e:
            logging.error("Error occured during model training.")
            raise CustomException(e, sys)