"""
api.py — FastAPI server for SNN Alzheimer's classifier.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /predict   — upload an MRI image, get prediction + anomaly flag
    GET  /health    — liveness check
    GET  /labels    — list class names
"""

import io
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predict import predict

IMAGE_SIZE = 128

app = FastAPI(
    title="Neuroimaging SNN Diagnostics API",
    description="Alzheimer's disease classification using Spiking Neural Network encoding + Random Forest.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LABELS = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]


# ── Response schema ───────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    anomaly_score: float
    is_anomaly: bool
    spike_mean: float
    warning: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/labels")
def get_labels():
    return {"labels": LABELS}


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Upload a grayscale MRI image (JPG/PNG) and get an Alzheimer's stage prediction.

    Returns:
        - prediction: class index (0-3)
        - label: class name
        - anomaly_score: z-score deviation from training distribution
        - is_anomaly: True if the image looks suspicious / out-of-distribution
        - spike_mean: mean spike activity for this image
        - warning: optional message if the image is flagged as anomalous
    """
    # Validate file type
    if file.content_type not in {"image/jpeg", "image/png", "image/bmp"}:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG."
        )

    # Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. File may be corrupt.")

    # Resize to expected input
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # Run prediction pipeline
    result = predict(img)

    # Add human-readable warning for anomalous inputs
    warning = None
    if result["is_anomaly"]:
        warning = (
            f"This image looks out-of-distribution (anomaly score: {result['anomaly_score']:.2f}). "
            "The prediction may be unreliable. Please verify the scan quality."
        )

    return PredictionResponse(**result, warning=warning)