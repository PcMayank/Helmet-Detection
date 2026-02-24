# 🪖 YOLOv8 Helmet Detection System

A real-time Helmet Detection system built using YOLOv8 and FastAPI.  
This project demonstrates an end-to-end Computer Vision pipeline including dataset preparation, model training, inference, REST API integration, and Docker deployment.

---

## 🚀 Features

- Real-time helmet detection using YOLOv8
- Custom dataset training (YOLO format)
- Image and webcam inference
- REST API using FastAPI
- Docker container support
- Modular training pipeline

---

## 📁 Project Structure

```
helmet-detection/
│
├── app/                # FastAPI backend
├── data/               # Dataset (images & labels)
├── model/              # Trained model (best.pt)
├── notebooks/          # Training notebook
├── src/                # Training & detection scripts
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/PcMayank/Helmet-Detection.git
cd Helmet-Detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Weights

Place trained model inside:

```
model/best.pt
```

---

## 🏋️ Model Training

```bash
python src/train.py --data data/data.yaml --epochs 50 --img-size 640
```

Or use notebook:

```
notebooks/helmet_detection_yolov8.ipynb
```

---

## 🔎 Image Inference

```python
from ultralytics import YOLO

model = YOLO("model/best.pt")
results = model.predict("image.jpg", conf=0.5)
results.show()
```

---

## 🌐 Run FastAPI Server

```bash
uvicorn app.app:app --reload
```

API Endpoint:

```
POST /predict
```

Test:

```bash
curl -X POST -F "file=@sample.jpg" http://localhost:8000/predict
```

---

## 🐳 Docker

Build:

```bash
docker build -t helmet-detector .
```

Run:

```bash
docker run -p 8000:8000 helmet-detector
```

---

## 📌 Resume Highlights

- Implemented YOLOv8 object detection model
- Built REST API using FastAPI
- Containerized application with Docker
- Designed ML training pipeline

---

## 👨‍💻 Author

Mayank