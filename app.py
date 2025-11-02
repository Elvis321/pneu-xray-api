# app.py
import os
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app_bootstrap import ensure_model_present, MODEL_PATH

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# --------------------------------------------------------------------------------------
# App + CORS
# --------------------------------------------------------------------------------------
app = FastAPI(title="Pneumonia X-Ray API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------------
# Config via env
# --------------------------------------------------------------------------------------
NUM_CLASSES = int(os.getenv("XRAY_NUM_CLASSES", "2"))  # 2 (Normal vs Pneumonia)
CLASS_NAMES = os.getenv("XRAY_CLASS_NAMES", "normal,pneumonia").split(",")
CLASS_NAMES = [c.strip() for c in CLASS_NAMES if c.strip()]
if len(CLASS_NAMES) != NUM_CLASSES:
    # Fallback safe labels
    CLASS_NAMES = [f"class_{i}" for i in range(NUM_CLASSES)]

# Force CPU (Render free tier / portability); switch to CUDA if you add GPU support
DEVICE = torch.device("cpu")

# ImageNet normalization for DenseNet
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
    # weights=None to avoid downloading weights in restricted envs
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

def _is_state_dict(obj) -> bool:
    """Heuristic: a mapping of tensor keys (and possibly 'state_dict' key)."""
    if isinstance(obj, dict):
        # Lightning checkpoints often have 'state_dict'
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return True
        # Vanilla state_dict: many keys with dots like 'features.denseblock1...'
        tensor_like = [k for k, v in obj.items() if isinstance(k, str)]
        return len(tensor_like) > 0
    return False

def _load_model_from_path(path: str, num_classes: int) -> nn.Module:
    """
    Tries:
      1) load full model (torch.save(model, ...))
      2) load state_dict (torch.save(model.state_dict(), ...))
      3) load lightning-like {'state_dict': ...}
    """
    with open(path, "rb") as f:
        blob = f.read()

    # First try: full model
    try:
        model = torch.load(io.BytesIO(blob), map_location=DEVICE)
        if isinstance(model, nn.Module):
            model.eval()
            return model
        # If it's not a Module, maybe it's a state dict-like
    except Exception as e_full:
        # fall through to state_dict logic
        pass

    # Second try: state_dict
    try:
        ckpt = torch.load(io.BytesIO(blob), map_location=DEVICE)
        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        elif _is_state_dict(ckpt):
            sd = ckpt
        else:
            raise RuntimeError("Checkpoint is neither a torch.nn.Module nor a recognizable state_dict")

        model = _build_densenet(num_classes)
        # Some checkpoints prefix keys (e.g., 'model.' or 'net.')
        # Try clean load; if it fails, strip common prefixes.
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            # Try stripping 'model.' prefix
            trimmed = {k.split("model.", 1)[-1] if k.startswith("model.") else k: v for k, v in sd.items()}
            model.load_state_dict(trimmed, strict=False)

        model.to(DEVICE).eval()
        return model
    except Exception as e_sd:
        raise RuntimeError(f"Model not available: {e_sd}") from e_sd

# Ensure model file exists (downloads if needed via app_bootstrap)
ensure_model_present()
Model: Optional[nn.Module] = _load_model_from_path(MODEL_PATH, NUM_CLASSES).to(DEVICE).eval()

# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------
class PredictResp(BaseModel):
    ok: bool
    predicted_class: str
    class_index: int
    probabilities: List[float]
    logits: List[float]

class HealthResp(BaseModel):
    status: str
    model_path: str
    num_classes: int
    class_names: List[str]

# --------------------------------------------------------------------------------------
# Helpers
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

    # If binary with 1 logit, treat as sigmoid; else softmax
    if logits.ndim == 0 or logits.shape == torch.Size([]):  # scalar
        probs = torch.sigmoid(logits).unsqueeze(0)
        # Convert to 2-class style: [1-p, p]
        probs = torch.stack([1.0 - probs, probs], dim=0)
        logits = torch.stack([-logits, logits], dim=0)
    elif logits.shape[-1] == 1 and NUM_CLASSES == 2:
        probs1 = torch.sigmoid(logits[:, 0])
        probs = torch.stack([1.0 - probs1, probs1], dim=1)[0]
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
        class_names=CLASS_NAMES,
    )

@app.post("/predict/xray", response_model=PredictResp)
async def predict_xray(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not content or len(content) < 100:
            raise HTTPException(status_code=400, detail="Uploaded file seems empty or too small.")
        out = _infer(content)
        cls_idx = out["cls_idx"]
        return PredictResp(
            ok=True,
            predicted_class=CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else str(cls_idx),
            class_index=cls_idx,
            probabilities=out["probs"],
            logits=out["logits"],
        )
    except HTTPException:
        raise
    except Exception as e:
        # Surface a concise error but keep 500
        raise HTTPException(status_code=500, detail=f"inference_error: {e}")

# Optional: local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=bool(os.getenv("DEV_RELOAD", "0") == "1"),
    )
