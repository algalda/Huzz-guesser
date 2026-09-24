import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from . import DEFAULT_MODEL_PATH, class_summary, predict_image, train_model

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "data" / "custom-dataset"
UI_ROOT = Path(__file__).parent
app = FastAPI(title="Huzz guesser")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def home() -> FileResponse:
    return FileResponse(UI_ROOT / "UI.html")


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse(UI_ROOT / "styles.css")


@app.get("/classes")
def classes() -> dict[str, list[dict[str, object]]]:
    return {"classes": class_summary(DATASET_ROOT)}


@app.post("/train")
async def train(dataset: UploadFile = File(...), epochs: int = Form(10)) -> dict[str, object]:
    if not dataset.filename or not dataset.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload a .zip file with one folder per class.")
    DATASET_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
    DATASET_ROOT.mkdir(parents=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as archive:
        archive.write(await dataset.read())
        archive_path = Path(archive.name)
    try:
        with zipfile.ZipFile(archive_path) as zip_file:
            for member in zip_file.infolist():
                destination = (DATASET_ROOT / member.filename).resolve()
                if DATASET_ROOT.resolve() not in destination.parents:
                    raise HTTPException(status_code=400, detail="The dataset archive contains an unsafe path.")
            zip_file.extractall(DATASET_ROOT)
        folders = list(DATASET_ROOT.iterdir())
        if len(folders) == 1 and folders[0].is_dir():
            dataset_dir = folders[0]
        else:
            dataset_dir = DATASET_ROOT
        return train_model(dataset_dir, epochs=max(1, min(100, epochs)))
    finally:
        archive_path.unlink(missing_ok=True)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if not DEFAULT_MODEL_PATH.exists():
        raise HTTPException(status_code=409, detail="Train a model before classifying images.")
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        return predict_image(image)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not classify this image: {error}") from error