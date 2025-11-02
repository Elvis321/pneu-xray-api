FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# (Optional) speedier builds for torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
# If you want to ship the checkpoint inside the image:
# COPY models/densnet_pneu_best.pt models/densnet_pneu_best.pt

ENV PORT=8080
# Local path OR HTTPS URL:
ENV XRAY_MODEL_PATH=models/densnet_pneu_best.pt
ENV CORS_ALLOW_ORIGINS=*

EXPOSE 8080
CMD ["python", "app.py"]
