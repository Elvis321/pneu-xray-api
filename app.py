import os, io, hashlib, tempfile, base64, requests
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models as tvm

# --------- Config ---------
MODEL_PATH = os.getenv("XRAY_MODEL_PATH", "models/densnet_pneu_best.pt")
ALLOWED_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
DEVICE = "cpu"  # simple & portable; switch to "cuda" if your host has a GPU
PORT = int(os.getenv("PORT", "8080"))

# --------- App & CORS ---------
app = FastAPI(title="Pneumonia Xray API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- I/O Schemas ---------
class PredictResp(BaseModel):
    probability: float
    label: str
    num_classes: int

# --------- Utils ---------
def _http_or_local_bytes(path_or_url: str) -> bytes:
    if path_or_url.startswith(("http://", "https://")):
        r = requests.get(path_or_url, timeout=120)
        if r.status_code != 200:
            raise FileNotFoundError(f"HTTP {r.status_code} when fetching {path_or_url}")
        return r.content
    with open(path_or_url, "rb") as f:
        return f.read()

def _ensure_model_file(path_or_url: str) -> str:
    """Download to /tmp once if it's a URL; return local file path."""
    if not path_or_url.startswith(("http://", "https://")):
        return path_or_url
    key = hashlib.sha256(path_or_url.encode()).hexdigest()[:24]
    dst = os.path.join(tempfile.gettempdir(), f"xray_{key}.pt")
    if not os.path.exists(dst):
        raw = _http_or_local_bytes(path_or_url)
        with open(dst, "wb") as f:
            f.write(raw)
    return dst

def _build_densenet(num_classes: int) -> nn.Module:
    m = tvm.densenet121(weights=None)
    in_feats = m.classifier.in_features
    m.classifier = nn.Linear(in_feats, num_classes)
    return m

def _infer_num_classes_from_state_dict(sd: dict) -> int:
    # Try common keys for DenseNet classifier
    for k in ("classifier.weight", "module.classifier.weight", "model.classifier.weight"):
        if k in sd and sd[k].ndim == 2:
            return sd[k].shape[0]
    # Fallback: 1
    return 1

# --------- Model load ---------
XRAY_MODEL: Optional[nn.Module] = None
XRAY_NUM_CLASSES: int = 1

def _load_model() -> None:
    global XRAY_MODEL, XRAY_NUM_CLASSES
    local_path = _ensure_model_file(MODEL_PATH)
    ckpt = torch.load(local_path, map_location="cpu")

    # ckpt can be a state_dict or a dict with 'state_dict'/'model'/...
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("state_dict", "model", "net")):
        state_dict = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("net")
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format")

    XRAY_NUM_CLASSES = _infer_num_classes_from_state_dict(state_dict)
    model = _build_densenet(XRAY_NUM_CLASSES)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(DEVICE)
    XRAY_MODEL = model

# Preprocessing (ImageNet stats; works for DenseNet pretrained pipelines)
TFM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def _pil_from_any(img_bytes: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # handles grayscale & PNG with alpha
    return im

def _predict_pneumonia(img_bytes: bytes) -> PredictResp:
    if XRAY_MODEL is None:
        try:
            _load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not available: {e}")

    pil = _pil_from_any(img_bytes)
    x = TFM(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = XRAY_MODEL(x)

    if logits.ndim == 2 and logits.shape[1] == 1:
        # single logit -> sigmoid
        prob = torch.sigmoid(logits)[0, 0].item()
        label = "pneumonia" if prob >= 0.5 else "normal"
        return PredictResp(probability=float(prob), label=label, num_classes=1)

    if logits.ndim == 2 and logits.shape[1] >= 2:
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        # Assume class index 1 = pneumonia if 2 classes; otherwise choose max
        if logits.shape[1] == 2:
            prob = float(probs[1])
            label = "pneumonia" if prob >= 0.5 else "normal"
        else:
            idx = int(np.argmax(probs))
            prob = float(probs[idx])
            label = f"class_{idx}"
        return PredictResp(probability=prob, label=label, num_classes=logits.shape[1])

    raise HTTPException(status_code=500, detail="Unexpected model output shape.")

# --------- Routes ---------
@app.get("/healthz")
def healthz():
    ok = True
    try:
        if XRAY_MODEL is None:
            _load_model()
    except Exception:
        ok = False
    return {"ok": ok, "model_loaded": XRAY_MODEL is not None, "num_classes": XRAY_NUM_CLASSES}

@app.post("/predict/xray", response_model=PredictResp)
async def predict_xray(
    file: UploadFile | None = File(default=None, description="X-ray image file"),
    image_base64: str | None = Form(default=None, description="Base64-encoded image"),
):
    """
    Either upload `file` (multipart/form-data) or send `image_base64` in a form field.
    """
    if file is None and image_base64 is None:
        raise HTTPException(status_code=400, detail="Provide file OR image_base64")

    try:
        if file is not None:
            img_bytes = await file.read()
        else:
            img_bytes = base64.b64decode(image_base64, validate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image input: {e}")

    return _predict_pneumonia(img_bytes)

# Entrypoint (for Docker/Render)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
