from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_BASE_MODEL = Path("yolo11n-cls.pt")
DEFAULT_MODEL_PATH = Path("models/model.pt")


def class_summary(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Return the labeled folders and image counts in a classification dataset."""
    root = Path(dataset_dir)
    return [
        {
            "name": folder.name,
            "images": sum(1 for item in folder.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS),
        }
        for folder in sorted(root.iterdir())
        if folder.is_dir()
    ] if root.exists() else []


def train_model(
    dataset_dir: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    base_model: str | Path = DEFAULT_BASE_MODEL,
    epochs: int = 10,
    image_size: int = 224,
) -> dict[str, Any]:
    """Train a classification model from folders whose names are the labels."""
    classes = class_summary(dataset_dir)
    if len(classes) < 2:
        raise ValueError("Add at least two labeled folders before training.")
    if any(item["images"] == 0 for item in classes):
        raise ValueError("Every labeled folder needs at least one image.")

    model = YOLO(str(base_model))
    metrics = model.train(
        data=str(dataset_dir),
        epochs=epochs,
        imgsz=image_size,
        plots=False,
        project=str(Path(model_path).parent / "training-runs"),
        name="template",
        exist_ok=True,
    )
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    best_weights = Path(metrics.save_dir) / "weights" / "best.pt"
    best_weights.replace(destination)
    return {"model": str(destination), "classes": classes, "top1": float(metrics.top1)}


def predict_image(image: Image.Image | str | Path, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Return the ranked class predictions for one image."""
    model = YOLO(str(model_path))
    result = model(image, verbose=False)[0]
    predictions = [
        {"label": result.names[int(class_id)], "confidence": float(confidence)}
        for class_id, confidence in zip(result.probs.top5, result.probs.top5conf)
    ]
    return {
        "label": predictions[0]["label"],
        "confidence": predictions[0]["confidence"],
        "predictions": predictions,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train and serve your own image classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("dataset", type=Path)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    subparsers.add_parser("serve")
    args = parser.parse_args()
    if args.command == "train":
        print(train_model(args.dataset, model_path=args.model, epochs=args.epochs))
    else:
        import uvicorn

        uvicorn.run("italian_brainrot_image_classifier.server:app", host="127.0.0.1", port=8000, reload=False)