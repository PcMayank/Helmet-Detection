import os
import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_training_curves(csv_path, output_dir):

    if not os.path.exists(csv_path):
        print(" results.csv not found")
        return

    df = pd.read_csv(csv_path)

    ensure_dir(output_dir)

    # Loss
    plt.figure()

    if "train/box_loss" in df:
        plt.plot(df["epoch"], df["train/box_loss"], label="Box Loss")

    if "train/cls_loss" in df:
        plt.plot(df["epoch"], df["train/cls_loss"], label="Cls Loss")

    plt.legend()
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

    # Accuracy
    plt.figure()

    if "metrics/mAP50(B)" in df:
        plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")

    if "metrics/mAP50-95(B)" in df:
        plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")

    plt.legend()
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Score")

    plt.savefig(os.path.join(output_dir, "accuracy.png"))
    plt.close()

    print(" Training graphs saved")