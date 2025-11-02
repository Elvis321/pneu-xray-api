# app_bootstrap.py
import os, hashlib, sys
from pathlib import Path

MODEL_PATH = os.getenv("XRAY_MODEL_PATH", "models/densnet_pneu_best.pt")
GDRIVE_ID  = os.getenv("XRAY_MODEL_GDRIVE_ID")  # the FILE id
SHA256     = os.getenv("XRAY_MODEL_SHA256")     # optional integrity check

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_model_present():
    p = Path(MODEL_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)

    # If exists, optionally verify hash
    if p.exists():
        if SHA256:
            got = _sha256(p)
            if got.lower() != SHA256.lower():
                print(f"[bootstrap] Hash mismatch. Expected {SHA256}, got {got}. Re-downloading…", file=sys.stderr)
                p.unlink(missing_ok=True)
            else:
                print(f"[bootstrap] Model present with valid SHA256: {got}")
                return
        else:
            print(f"[bootstrap] Model already present at {p} (size={p.stat().st_size} bytes)")
            return

    if not GDRIVE_ID:
        raise RuntimeError("Model file missing and XRAY_MODEL_GDRIVE_ID not set.")

    # Use gdown which handles Google Drive confirmation cookies for large files
    import gdown
    url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
    print(f"[bootstrap] Downloading model from Google Drive -> {p}")
    out = gdown.download(url=url, output=str(p), quiet=False, fuzzy=True)
    if not out or not p.exists() or p.stat().st_size < 1024:
        raise RuntimeError("Download seems invalid (too small). Check the file ID & sharing perms (Anyone with link: Viewer).")

    if SHA256:
        got = _sha256(p)
        if got.lower() != SHA256.lower():
            raise RuntimeError(f"Downloaded file hash mismatch. Expected {SHA256}, got {got}")

    # Quick binary sanity check: read first bytes (shouldn’t start with '<!DOCTYPE' or 'PK' etc.)
    with p.open("rb") as f:
        head = f.read(32)
    if head.startswith(b"<!") or head.startswith(b"<html") or head.startswith(b"PK"):
        raise RuntimeError("Downloaded HTML/ZIP instead of a .pt checkpoint. Double-check XRAY_MODEL_GDRIVE_ID.")
    print(f"[bootstrap] Model downloaded. Size={p.stat().st_size} bytes")
