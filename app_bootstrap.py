# app_bootstrap.py
import os, io, sys, re, hashlib
from pathlib import Path

MODEL_PATH = os.getenv("XRAY_MODEL_PATH", "models/densnet_pneu_best.pt")
MODEL_URL  = os.getenv("XRAY_MODEL_URL", "")          # e.g. https://.../densnet_pneu_best.pt
GDRIVE_ID  = os.getenv("XRAY_MODEL_GDRIVE_ID", "")    # e.g. 1AbCdefGhIJ...

_MIN_BYTES = int(os.getenv("XRAY_MODEL_MIN_BYTES", "1048576"))  # 1 MB sanity

def _read_head(path, n=32):
    with open(path, "rb") as f:
        return f.read(n)

def _looks_like_torch_checkpoint(head: bytes) -> bool:
    # Newer torch saves are ZIP: starts with b"PK\x03\x04"
    # Pickle binary protocol usually starts with \x80\x05 or similar
    return head.startswith(b"PK\x03\x04") or head.startswith(b"\x80")

def _explain_bad_header(head: bytes) -> str:
    # Try to guess common mistakes
    htxt = head.decode("latin1", errors="ignore")
    if "<!DOCTYPE html" in htxt or "<html" in htxt.lower():
        return "It looks like an HTML page (likely a Drive/GitHub download page)."
    if "version https://git-lfs.github.com/spec" in htxt:
        return "It is a Git LFS pointer file, not the binary model."
    if htxt[:1] == "\r" or htxt[:1] == "\n":
        return "File begins with a newline/carriage return — likely text-mode or corrupted download."
    return "Unrecognized header — file is likely not a PyTorch checkpoint."

def _download_http(url: str, path: str):
    import requests
    Path(Path(path).parent).mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def _download_gdrive(file_id: str, path: str):
    import gdown
    Path(Path(path).parent).mkdir(parents=True, exist_ok=True)
    # gdown handles confirm tokens automatically
    gdown.download(id=file_id, output=path, quiet=False)

def ensure_model_present():
    """Ensure the model file exists locally and looks valid."""
    p = Path(MODEL_PATH)
    if not p.exists():
        if GDRIVE_ID:
            _download_gdrive(GDRIVE_ID, MODEL_PATH)
        elif MODEL_URL:
            _download_http(MODEL_URL, MODEL_PATH)
        else:
            raise RuntimeError(
                f"Model not available: {MODEL_PATH} not found and neither XRAY_MODEL_GDRIVE_ID "
                f"nor XRAY_MODEL_URL env vars are set."
            )

    # Sanity checks
    size = p.stat().st_size
    if size < _MIN_BYTES:
        raise RuntimeError(
            f"Model not available: file too small ({size} bytes) at {MODEL_PATH}. "
            f"This usually means you downloaded an HTML page or LFS pointer. "
            f"Set XRAY_MODEL_GDRIVE_ID or XRAY_MODEL_URL to the *raw* model."
        )

    head = _read_head(MODEL_PATH, 32)
    if not _looks_like_torch_checkpoint(head):
        hint = _explain_bad_header(head)
        # show first ~60 printable chars for debugging
        preview = head.decode("latin1", errors="ignore")
        preview = re.sub(r"[\x00-\x1F\x7F]+", " ", preview)[:60]
        raise RuntimeError(
            f"Model not available: invalid header. {hint}\n"
            f"First bytes preview: '{preview}'\n"
            f"Path: {MODEL_PATH}"
        )

def debug_read(path: str):
    """Helper you can run in a one-off Python shell to inspect a file head/size quickly."""
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    b = _read_head(path, 64)
    return {
        "exists": True,
        "size": p.stat().st_size,
        "head_hex": b[:16].hex(),
        "head_printable": re.sub(r"[^\x20-\x7E]+", " ", b.decode('latin1', 'ignore'))[:80],
        "torch_like": _looks_like_torch_checkpoint(b),
    }
