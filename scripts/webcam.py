from src.helmet.components.webcam import WebcamDetector

if __name__ == "__main__":
    print("Starting webcam...")

    model_path = "runs/detect/runs/exp/weights/best.pt"

    detector = WebcamDetector(model_path)
    detector.run()