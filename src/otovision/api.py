from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from fastapi.responses import JSONResponse

from .inference import load_checkpoint, predict_image

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", "/models/best.pt")
    try:
        STATE["bundle"] = load_checkpoint(model_path)
        STATE["error"] = None
    except Exception as exc:  # service can still expose health reason
        STATE["bundle"] = None
        STATE["error"] = repr(exc)
    yield
    STATE.clear()


app = FastAPI(
    title="OtoVision-MLOps API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    ready = STATE.get("bundle") is not None
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "model_loaded": ready},
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if STATE.get("bundle") is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    payload = await file.read(10 * 1024 * 1024 + 1)
    if len(payload) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MiB.")
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unreadable image.") from exc
    model, class_names, image_size, device = STATE["bundle"]
    return predict_image(model, image, class_names, image_size, device)
