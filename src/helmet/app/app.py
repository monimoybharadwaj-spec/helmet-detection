from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import shutil
import os

app = FastAPI(title="Helmet Detection API")

MODEL_PATH = os.path.join("runs", "detect", "runs", "exp", "weights", "best.pt")
model = YOLO(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "Helmet Detection API is running. Go to /docs"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run inference
        results = model.predict(file_path, conf=0.6)

        boxes = results[0].boxes

        helmet_detected = False
        no_helmet_detected = False

        detections = []

        for box in boxes:
            conf = float(box.conf)
            cls = int(box.cls)

            class_name = model.names[cls]

            # Store detections
            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": box.xyxy.tolist()
            })

            # Decision logic
            if conf > 0.6:
                if class_name == "helmet":
                    helmet_detected = True
                elif class_name == "no-helmet":
                    no_helmet_detected = True

        # Final logic
        if helmet_detected:
            status = "Helmet Detected"
        elif no_helmet_detected:
            status = "No Helmet"
        else:
            status = "No Detection"

        # Cleanup temp file
        os.remove(file_path)

        return {
            "status": status,
            "detections": detections
        }

    except Exception as e:
        return {"error": str(e)}