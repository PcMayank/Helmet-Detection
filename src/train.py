import os
import argparse
import shutil
from ultralytics import YOLO
from utils import ensure_dir, plot_training_curves


def train_model(data_yaml, epochs=50, img_size=640, weights="yolov8s.pt"):

    print(" Loading YOLO model...")
    model = YOLO(weights)

    print(" Training started...")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        patience=15,
        amp=True
    )

    exp_dir = results.save_dir
    print(f" Training completed: {exp_dir}")

    # Plot training curves
    csv_path = os.path.join(exp_dir, "results.csv")
    plot_training_curves(csv_path, exp_dir)

    # Copy best model
    best_weight = os.path.join(exp_dir, "weights", "best.pt")

    target_dir = "../model"
    ensure_dir(target_dir)

    if os.path.exists(best_weight):
        shutil.copy(best_weight, os.path.join(target_dir, "best.pt"))
        print(" Best model saved in model/best.pt")
    else:
        print(" best.pt not found!")

    return exp_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="../data/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--weights", default="yolov8s.pt")

    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        img_size=args.img,
        weights=args.weights
    )