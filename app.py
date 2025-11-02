# app.py
import os
from typing import Optional, List
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app_bootstrap import ensure_model_present, MODEL_PATH

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --------------------------------------------------------------------------------------
# App + CORS
# --------------------------------------------------------------------------------------
app = FastAPI(title="Pneumonia X-Ray API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------------
# Config via env
# --------------------------------------------------------------------------------------
NUM_CLASSES = int(os.getenv("XRAY_NUM_CLASSES", "2"))  # change if your head differs
DEVICE = torch.device("cpu")

IMG_SIZE = int(os.getenv("XRAY_IMG_SIZE", "224"))
_xforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --------------------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------------------
def _build_densenet(num_classes: int) -> nn.Module:
    # weights=None => no external downloads in restricted envs
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

def _is_state_dict(obj) -> bool:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return True
        # vanilla state_dict heuristic
        return any(isinstance(k, str) for k in obj.keys())
    return False

def _load_model_from_path(path: str, num_classes: int) -> nn.Module:
    with open(path, "rb") as f:
        blob = f.read()

    # 1) try full model
    try:
        mdl = torch.load(io.BytesIO(blob), map_location=DEVICE)
        if isinstance(mdl, nn.Module):
            mdl.eval()
            return mdl
    except Exception:
        pass

    # 2) try state_dict
    ckpt = torch.load(io.BytesIO(blob), map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif _is_state_dict(ckpt):
        sd = ckpt
    else:
        raise RuntimeError("Checkpoint is neither a torch.nn.Module nor a recognizable state_dict")

    mdl = _build_densenet(num_classes)
    try:
        mdl.load_state_dict(sd, strict=False)
    except Exception:
        # strip common prefixes (e.g., 'model.')
        trimmed = {k.split("model.", 1)[-1] if k.startswith("model.") else k: v for k, v in sd.items()}
        mdl.load_state_dict(trimmed, strict=False)

    mdl.to(DEVICE).eval()
    return mdl

# Ensure model file is present (download if your app_bootstrap does that)
ensure_model_present()
Model: Optional[nn.Module] = _load_model_from_path(MODEL_PATH, NUM_CLASSES).to(DEVICE).eval()

# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------
class PredictResp(BaseModel):
    ok: bool
    class_index: int
    probabilities: List[float]
    logits: List[float]

class HealthResp(BaseModel):
    status: str
    model_path: str
    num_classes: int

# --------------------------------------------------------------------------------------
# Inference helper
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def _infer(img_bytes: bytes):
    if Model is None:
        raise RuntimeError("Model not loaded")

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = _xforms(img).unsqueeze(0).to(DEVICE)

    logits = Model(x)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    logits = logits.squeeze(0)

    # Binary special-casing: if only a single logit is produced, map to 2-class probs
    if logits.ndim == 0 or logits.shape == torch.Size([]):
        p1 = torch.sigmoid(logits)
        probs = torch.stack([1 - p1, p1], dim=0)
        logits = torch.stack([-logits, logits], dim=0)
    elif logits.shape[-1] == 1 and NUM_CLASSES == 2:
        p1 = torch.sigmoid(logits[:, 0])
        probs = torch.stack([1 - p1, p1], dim=1)[0]
        logits = torch.stack([-logits[:, 0], logits[:, 0]], dim=1)[0]
    else:
        probs = torch.softmax(logits, dim=-1)

    cls_idx = int(torch.argmax(probs).item())
    return {
        "logits": logits.detach().cpu().tolist(),
        "probs": probs.detach().cpu().tolist(),
        "cls_idx": cls_idx,
    }

# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.get("/health", response_model=HealthResp)
def health():
    return HealthResp(
        status="ok",
        model_path=MODEL_PATH,
        num_classes=NUM_CLASSES,
    )

@app.post("/predict/xray", response_model=PredictResp)
async def predict_xray(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not content or len(content) < 100:
            raise HTTPException(status_code=400, detail="Uploaded file seems empty or too small.")
        out = _infer(content)
        return PredictResp(
            ok=True,
            class_index=out["cls_idx"],
            probabilities=out["probs"],
            logits=out["logits"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference_error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=bool(os.getenv("DEV_RELOAD", "0") == "1"),
    )
