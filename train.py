"""
train.py — Retrain the detection model locally.

This is NOT used by the deployed Flask app (which runs ONNX models from
models/). Use this only if you want to retrain on new/expanded data —
for example, to finally add 'bottle', 'screw', 'headset', 'spectacles'
which the current models have never been trained on (see README).

Setup (local machine only — do not run this on Vercel):
    pip install -r requirements-train.txt
    python train.py --model yolov8s --epochs 100

After training, export to ONNX and drop the result into models/:
    (this script does the export for you automatically at the end)
"""

import argparse
import os
import shutil


def get_device() -> str:
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def train(args):
    from ultralytics import YOLO

    device = get_device()
    print(f"\n{'='*55}\n  SecureScan — Retraining\n  Model : {args.model}.pt"
         f"\n  Epochs: {args.epochs}\n  Device: {device}\n{'='*55}\n")

    model = YOLO(f"{args.model}.pt")
    model.train(
        data=args.data, epochs=args.epochs, imgsz=640, batch=args.batch,
        device=device, project="runs/detect", name=args.name, patience=30,
        degrees=10, translate=0.1, scale=0.4, shear=2.0, perspective=0.0005,
        flipud=0.3, fliplr=0.5, mosaic=0.8, mixup=0.1, copy_paste=0.05,
        hsv_h=0.0, hsv_s=0.3, hsv_v=0.4,
    )

    best_pt = f"runs/detect/{args.name}/weights/best.pt"
    if os.path.exists(best_pt):
        print("\nExporting to ONNX…")
        YOLO(best_pt).export(format="onnx", simplify=True, dynamic=False, imgsz=640)
        onnx_path = best_pt.replace(".pt", ".onnx")
        dest = os.path.join("models", f"{args.deploy_as}.onnx")
        shutil.copy(onnx_path, dest)
        print(f"Copied to {dest} — update inference.py MODEL_CONFIGS with its class list,")
        print("then redeploy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain the SecureScan detection model")
    parser.add_argument("--model", default="yolov8s",
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
    parser.add_argument("--data", default="dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="train_new")
    parser.add_argument("--deploy-as", default="model_a",
                        help="filename (without .onnx) to copy the exported model to in models/")
    train(parser.parse_args())
