from src.helmet.components.predictor import Predictor

if __name__ == "__main__":
    model_path = "runs/detect/runs/exp/weights/best.pt"

    predictor = Predictor(model_path)

    # Testing on sample image
    predictor.predict(source = "data/helmet_detection/test/images")