# app_bootstrap.py
import os
from pathlib import Path

MODEL_PATH = os.getenv("XRAY_MODEL_PATH", "models/densnet_pneu_best.pt")
GDRIVE_ID  = os.getenv("XRAY_MODEL_GDRIVE_ID")  # e.g. 1AbCdEfGhIJklMNopQRstuVWxyz

def ensure_model_present():
    p = Path(MODEL_PATH)
    if p.exists():
        print(f"[bootstrap] Model already present at {p}")
        return
    if not GDRIVE_ID:
        raise RuntimeError("Model file missing and XRAY_MODEL_GDRIVE_ID not set.")

    # Download from Google Drive via gdown
    import gdown
    url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"[bootstrap] Downloading model from Google Drive -> {p}")
    gdown.download(url, str(p), quiet=False)
    print("[bootstrap] Model downloaded.")
