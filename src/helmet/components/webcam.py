import sys
import cv2
from ultralytics import YOLO
from src.helmet.logger.logger import logging
from src.helmet.exception.exception import CustomException


class WebcamDetector:

    def __init__(self, model_path: str):
        try:
            self.model = YOLO(model_path)
            logging.info(f"Loaded model from {model_path}")
        except Exception as e:
            raise CustomException(e, sys)

    def run(self):
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

            if not cap.isOpened():
                raise Exception("Webcam not accessible")

            logging.info("Starting webcam detection... Press 'q' to exit")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO inference
                results = self.model(frame, conf=0.6)

                boxes = results[0].boxes

                helmet_detected = False
                no_helmet_detected = False

                # Draw bounding boxes
                annotated_frame = results[0].plot()

                for box in boxes:
                    conf = float(box.conf)
                    cls = int(box.cls)

                    if conf > 0.6:
                        class_name = self.model.names[cls]

                        if class_name == "helmet":
                            helmet_detected = True

                        elif class_name == "no-helmet":
                            no_helmet_detected = True

                if helmet_detected:
                    label = "Helmet Detected"
                    color = (0, 255, 0)
                elif no_helmet_detected:
                    label = "No Helmet"
                    color = (0, 0, 255)
                else:
                    label = "No Detection"
                    color = (255, 255, 0)

                # Displaying labels
                cv2.putText(
                    annotated_frame,
                    label,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
                )

                # Show frame
                cv2.imshow("Helmet Detection (Press q to quit)", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            logging.error("Error in webcam detection")
            raise CustomException(e, sys)