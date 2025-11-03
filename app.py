# app.py
import os
import io
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

# If you used the little helper that ensures the model file exists:
# (keep these lines; remove if you chose not to use app_bootstrap)
from app_bootstrap import ensure_model_present, MODEL_PATH

# --------------------------- FastAPI app ---------------------------
app = FastAPI(title="Pneumonia X-ray API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # tighten this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Labels ---------------------------
PATTERN_KEYS = ["pattern_alveolar", "pattern_interstitial", "pattern_broncho"]
LOBE_KEYS    = ["loc_rul", "loc_rml", "loc_rll", "loc_lul", "loc_lll",
                "loc_right", "loc_left", "loc_bilateral"]
FALLBACK_LABELS = PATTERN_KEYS + LOBE_KEYS

# --------------------------- Model init ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model: torch.nn.Module = None
_label_names: List[str] = FALLBACK_LABELS

def _build_empty_model(num_labels: int) -> torch.nn.Module:
    try:
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        m = models.densenet121(weights=weights)
    except Exception:
        m = models.densenet121(pretrained=True)  # older torchvision fallback
    in_features = m.classifier.in_features
    m.classifier = nn.Linear(in_features, num_labels)
    return m

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _load_checkpoint(path: str) -> Tuple[torch.nn.Module, List[str]]:
    ck = torch.load(path, map_location=DEVICE)
    # try to fetch label names; otherwise fall back
    label_names = ck.get("label_names", FALLBACK_LABELS) if isinstance(ck, dict) else FALLBACK_LABELS
    num_labels = len(label_names)

    model = _build_empty_model(num_labels)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    return model, label_names

# preprocessing must mirror training
_mean = [0.485, 0.456, 0.406]
_std  = [0.229, 0.224, 0.225]
_infer_tfms = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(_mean, _std),
])

# --------------------------- I/O schemas ---------------------------
class TopProb(BaseModel):
    label: str
    prob: float

class PredictResp(BaseModel):
    top: List[TopProb]
    on_threshold: List[TopProb]
    raw: Dict[str, float]  # label -> prob

# For URL-based prediction
class PredictFromUrlReq(BaseModel):
    image_url: HttpUrl
    topk: int = 10
    thresh: float = 0.50

# --------------------------- Helpers ---------------------------
def _ensure_loaded():
    global _model, _label_names
    if _model is not None:
        return
    # make sure model file exists & is valid (handled by app_bootstrap checks)
    ensure_model_present()
    try:
        _model, _label_names = _load_checkpoint(MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not available: {e}")

def _fetch_image_bytes(url: str, timeout: float = 20.0) -> bytes:
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        content = r.content
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large")
        return content
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")

@torch.no_grad()
def _predict_pil(img: Image.Image, topk: int, thresh: float) -> PredictResp:
    x = _infer_tfms(img.convert("L")).unsqueeze(0).to(DEVICE)
    logits = _model(x).squeeze(0).detach().cpu().numpy()
    probs = _sigmoid_np(logits)

    order = np.argsort(-probs)[:topk]
    top = [TopProb(label=_label_names[i], prob=float(probs[i])) for i in order]

    on_idx = np.where(probs >= thresh)[0]
    on = [TopProb(label=_label_names[i], prob=float(probs[i])) for i in on_idx]

    raw = { _label_names[i]: float(probs[i]) for i in range(len(_label_names)) }

    return PredictResp(top=top, on_threshold=on, raw=raw)

# --------------------------- Routes ---------------------------
@app.get("/healthz")
def healthz():
    _ensure_loaded()
    return {"ok": True, "device": DEVICE, "labels": _label_names}

@app.get("/labels")
def labels():
    _ensure_loaded()
    return {"labels": _label_names}

# multipart upload (kept for curl/tests)
@app.post("/predict/xray", response_model=PredictResp)
async def predict_xray(
    file: UploadFile = File(..., description="Chest X-ray image (JPG/PNG/TIFF)"),
    topk: int = Query(10, ge=1, le=64),
    thresh: float = Query(0.50, ge=0.0, le=1.0)
):
    _ensure_loaded()
    try:
        content = await file.read()
        if not content:
            raise ValueError("Empty file.")
        img = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        return _predict_pil(img, topk=topk, thresh=thresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

# **NEW**: URL-based prediction
@app.post("/predict/xray/url", response_model=PredictResp)
async def predict_xray_from_url(req: PredictFromUrlReq):
    _ensure_loaded()
    content = _fetch_image_bytes(str(req.image_url))
    try:
        img = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        return _predict_pil(img, topk=req.topk, thresh=req.thresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

# --------------------------- Dev entrypoint ---------------------------
if __name__ == "__main__":
    # local dev: uvicorn app:app --reload --port 8080
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
