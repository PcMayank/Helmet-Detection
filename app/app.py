import os
import uuid
import shutil
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO


MODEL_PATH = "model/best.pt"
UPLOAD_DIR = "app/uploads"
RESULTS_DIR = "app/results"


os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


app = FastAPI(title="Helmet Detection API")

print(" Loading model...")
model = YOLO(MODEL_PATH)


@app.get("/")
def home():
    return {"status": "Helmet Detection API Running"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image"):
        raise HTTPException(400, "Only images allowed")

    file_id = str(uuid.uuid4())

    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = model.predict(input_path)

    detections = []

    r = results[0]

    for box in r.boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        detections.append({
            "class": model.names[cls],
            "confidence": round(conf, 3)
        })

    annotated = r.plot()

    result_name = f"{file_id}_out.jpg"
    result_path = os.path.join(RESULTS_DIR, result_name)

    cv2.imwrite(result_path, annotated)

    return {
        "detections": detections,
        "result_url": f"http://127.0.0.1:8000/download/{result_name}"
    }


@app.get("/download/{name}")
def download(name: str):

    path = os.path.join(RESULTS_DIR, name)

    if not os.path.exists(path):
        raise HTTPException(404, "File not found")

    return FileResponse(path)