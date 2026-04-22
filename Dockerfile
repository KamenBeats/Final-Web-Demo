# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

WORKDIR /app

RUN chmod 1777 /tmp
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        libgl1 libglx-mesa0 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf python3.10 /usr/bin/python && \
    ln -sf pip3 /usr/bin/pip

COPY requirements.txt .
# Cài PyTorch với CUDA 12.8 trước — cache lại
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache \
    pip install "torch>=2.0.1" --index-url https://download.pytorch.org/whl/cu128
# Install xformers from PyTorch index to match CUDA/torch version
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache \
    pip install xformers --index-url https://download.pytorch.org/whl/cu128 || \
    echo "xformers not available for this CUDA version, skipping"
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache \
    pip install -r requirements.txt

COPY . .

# ── Node.js for React frontend build ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Build React frontend at image build time
RUN cd react_app/frontend && npm install --production=false && npm run build

EXPOSE 7860

# ── Gradio mode (old) ─────────────────────────────────────────────────────────
# CMD ["python", "app.py"]

# ── React + FastAPI mode (new) — single port 7860 ─────────────────────────────
CMD ["python", "-m", "uvicorn", "react_app.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]