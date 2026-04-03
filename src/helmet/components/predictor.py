import os
import sys
from ultralytics import YOLO
from src.helmet.logger.logger import logging
from src.helmet.exception.exception import CustomException

class Predictor:

    def __init__(self, model_path : str):
        try:
            self.model = YOLO(model_path)
            logging.info(f"Loaded model from {model_path}.")

        except Exception as e:
            raise CustomException(e, sys)
        
    def predict(self, source: str, save: bool = True):
        try:
            results = self.model.predict(
                source = source,
                conf = 0.6,
                save = save
            )

            output = []

            for r in results:
                if len(r.boxes) > 0:
                    status = "Helmet Detected"
                else:
                    status = "No Helmet"

                detections = []
                for box in r.boxes:
                    detections.append({
                        "class": self.model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy.tolist()
                    })

                output.append({
                    "status": status,
                    "detections": detections
                })

            logging.info(f"Prediction completed from source: {source}")
            return output

        except Exception as e:
            logging.error("Error occured during prediction.")
            raise CustomException(e, sys)