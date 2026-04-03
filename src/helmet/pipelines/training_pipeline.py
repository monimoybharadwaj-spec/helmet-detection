import sys
from src.helmet.entity.config_entity import ModelTrainerConfig
from src.helmet.entity.artifact_entity import ModelTrainerArtifact
from src.helmet.components.model_trainer import ModelTrainer
from src.helmet.logger.logger import logging
from src.helmet.exception.exception import CustomException

class ModelTrainingPipeline:

    def __init__(self):   
        try:
            self.model_trainer_config = ModelTrainerConfig()
            logging.info("Training pipeline initialised.")
        except Exception as e:
            raise CustomException(e, sys)
        
    def pipeline_run(self):
        try:
            logging.info("Training pipeline starting...")
            trainer = ModelTrainer(self.model_trainer_config)

            # Start training
            model_trainer_artifact = trainer.model_training()
            logging.info(f"Model saved at {model_trainer_artifact.model_path}.")
            logging.info(f"Metrics saved at {model_trainer_artifact.metrics}.")
            return model_trainer_artifact
        
        except Exception as e:
            logging.error("Error occured.")
            raise CustomException(e, sys)