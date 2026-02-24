import os
import argparse
import random
import cv2
from ultralytics import YOLO
from utils import ensure_dir


def run_detection(weights, source, samples=10):

    print(" Loading model...")
    model = YOLO(weights)

    print(" Running inference...")

    results = model.predict(
        source=source,
        save=True,
        imgsz=640,
        conf=0.25
    )

    run_dir = results[0].save_dir
    print(" Results saved in:", run_dir)

    overlay_dir = os.path.join(run_dir, "overlays")
    ensure_dir(overlay_dir)

    selected = random.sample(results, min(samples, len(results)))

    for res in selected:
        img = res.plot()

        name = os.path.splitext(os.path.basename(res.path))[0]
        out_file = f"{name}_overlay.jpg"

        cv2.imwrite(os.path.join(overlay_dir, out_file), img)

    print(" Overlays saved in:", overlay_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", default="../model/best.pt")
    parser.add_argument("--source", default="../data/images/test")
    parser.add_argument("--samples", type=int, default=10)

    args = parser.parse_args()

    run_detection(
        args.weights,
        args.source,
        args.samples
    )