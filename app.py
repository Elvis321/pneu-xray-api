# app.py
import requests
from pydantic import BaseModel, HttpUrl
import os, io
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, AnyHttpUrl

import httpx  # for server-side fetch of image_url

# If you already created app_bootstrap.py earlier, keep it.
# It should define ensure_model_present() and MODEL_PATH (or you can just set XRAY_MODEL_PATH).
try:
    from app_bootstrap import ensure_model_present, MODEL_PATH
except Exception:
    ensure_model_present = None
    MODEL_PATH = os.getenv("XRAY_MODEL_PATH", "models/densnet_pneu_best.pt")

# --------------------------- FastAPI app ---------------------------
app = FastAPI(title="Pneumonia X-ray API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten in prod
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
_model: Optional[torch.nn.Module] = None
_label_names: List[str] = FALLBACK_LABELS

def _build_empty_model(num_labels: int) -> torch.nn.Module:
    # Torchvision new API first, fall back to old `pretrained=True`
    try:
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        m = models.densenet121(weights=weights)
    except Exception:
        m = models.densenet121(pretrained=True)
    in_features = m.classifier.in_features
    m.classifier = nn.Linear(in_features, num_labels)
    return m

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _load_checkpoint(path: str) -> Tuple[torch.nn.Module, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ck = torch.load(path, map_location=DEVICE)
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

class XrayURLIn(BaseModel):
    image_url: AnyHttpUrl
    return_all_labels: bool | None = False
    topk: int | None = 10
    thresh: float | None = 0.50

# --------------------------- Helpers ---------------------------
def _ensure_loaded():
    global _model, _label_names, MODEL_PATH
    if _model is not None:
        return
    # If you have bootstrap, use it; otherwise assume env path is set.
    if ensure_model_present:
        ensure_model_present()
    try:
        _model, _label_names = _load_checkpoint(MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not available: {e}")

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

# (A) multipart upload (works with Postman/curl)
@app.post("/predict/xray", response_model=PredictResp)
async def predict_xray(
    file: UploadFile = File(..., description="Chest X-ray image (JPG/PNG/TIFF)"),
    topk: int = Query(10, ge=1, le=64),
    thresh: float = Query(0.50, ge=0.0, le=1.0),
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

# (B) JSON body with image_url (ideal from Next.js with Firebase Storage URL)
@app.post("/predict/xray/url", response_model=PredictResp)
async def predict_xray_by_url(payload: XrayURLIn):
    _ensure_loaded()
    topk = payload.topk or 10
    thresh = payload.thresh or 0.50
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(str(payload.image_url))
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch image_url: {e}")

    try:
        return _predict_pil(img, topk=topk, thresh=thresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

# --------------------------- Dev entrypoint ---------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
