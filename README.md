# Helmet Detection using YOLOv8 + FastAPI + Docker

## Overview

This project implements an **end-to-end computer vision system** to detect whether a person is wearing a helmet using **YOLOv8**.

It includes:

* Model training using YOLOv8
* Data analysis and visualization
* FastAPI-based inference API
* Dockerized deployment

---

## Problem Statement

Detects helmet usage in real-world scenarios such as construction sites.

---

## Approach

1. Dataset preparation and annotation (YOLO format)
2. Data understanding and visualization
3. Model training using YOLOv8
4. Model evaluation and performance analysis
5. API development using FastAPI
6. Containerization using Docker
7. Deployment to cloud (Render)

---

## 📂 Project Structure

```
Helmet-Detection/
│
├── data/
│   └── helmet_detection/
│       ├── train/
│       ├── valid/
│       └── test/
│
├── notebooks/
│   ├── helmet_data_understanding.ipynb
│   ├── helmet_model_training_insights.ipynb
│   └── helmet_model_evaluation_insights.ipynb
│
├── runs/                 
│   └── detect/
│       └── runs/exp/
│           └── weights/
│               └── best.pt
│
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── webcam.py
│
├── src/helmet/
│   ├── app/            
│   ├── components/
│   ├── pipelines/
│   ├── utils/
│   ├── logger/
│   ├── exception/
│   └── constants/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Dataset

* Dataset follows YOLO format:

  ```
  class_id x_center y_center width height
  ```

* Structure:

  ```
  data/helmet_detection/
  ├── train/
  ├── valid/
  └── test/
  ```

* Classes:

  * `helmet`
  * `no-helmet`

> 📌  Dataset Note : The dataset was obtained from a publicly available source (e.g., Roboflow/Kaggle) and follows standard object detection annotation practices.

Due to licensing or availability constraints, the dataset is not included in this repository.

---

## Model Training

* Model used: **YOLOv8 (Ultralytics)**

* Variants tested:

  * YOLOv8s (Small)

* Training configuration:

  * Epochs: 10
  * Image size: 512
  * Batch size: 4

* Training executed via pipeline (not notebooks due to compute cost)

---

##  Results

* mAP@50: ~0.86
* mAP@50-95: ~0.51

### Observations:

* Strong detection performance
* Slight limitation in bounding box precision
* Model converges around 6–8 epochs

---

## 🧪 Notebooks

| Notebook                 | Description                             |
| ------------------------ | --------------------------------------- |
| Data Understanding       | Dataset analysis and class distribution |
|                          | and bounding box verification           |
| Training Experiments     | Model insights (analysis only)          |
| Model Evaluation         | Predictions analysis                    |

---

## API (FastAPI)

### Endpoints

#### `GET /`

Health check endpoint

#### `POST /predict`

Upload an image and get predictions

**Response Example:**

```json
{
  "status": "Helmet Detected",
  "detections": [
    {
      "class": "helmet",
      "confidence": 0.92,
      "bbox": [[x1, y1, x2, y2]]
    }
  ]
}
```

---

## Docker Setup

### Build image

```bash
docker build -t helmet-detection .
```

### Run container

```bash
docker run -p 8000:8000 helmet-detection
```

### Access API

```
http://localhost:8000/docs
```

---

## 🌐 Deployment

Deployed using **Render**

👉 Live API:

```
https://your-app-name.onrender.com/docs
```

*(Replace with your actual link)*

---

## 🧠 Key Learnings

* End-to-end ML system design
* YOLOv8 training and evaluation
* FastAPI backend integration
* Docker containerization
* Handling real-world edge cases

---

## Challenges

* Long training time (8+ hours)
* Dataset limitations (occlusion, lighting)
* Docker dependency size (PyTorch)

---

## Future Improvements

* Use YOLOv8m/l for better accuracy
* Improve dataset diversity
* Add real-time video inference
* Optimize Docker image size
* Add frontend UI

---

## Tech Stack

* Python
* YOLOv8 (Ultralytics)
* FastAPI
* OpenCV
* Docker
* Matplotlib / Pandas

---

## Author

**Monimoy Bharadwaj**

---

## If you like this project

Give it a ⭐ on GitHub!
# helmet-detection
