# CUDA 12.8 + PyTorch 2.8 base (RunPod official)
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_NO_TORCHVISION=1 \
    PIP_NO_CACHE_DIR=1

# System deps: nginx binary, git-lfs, curl, htpasswd
RUN apt-get update && apt-get install -y --no-install-recommends \
      nginx apache2-utils git git-lfs ca-certificates curl procps lsof \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Python deps
RUN python3 -m pip install --upgrade pip && \
    pip install \
      fastapi==0.115.0 uvicorn[standard]==0.30.6 pillow==10.4.0 requests==2.32.3 \
      accelerate==0.33.0 "bitsandbytes>=0.47.0" && \
    pip install \
      "git+https://github.com/huggingface/diffusers.git@main" \
      "git+https://github.com/huggingface/transformers.git@main"

# -------------------------
# App code (bundled into image)
# -------------------------
WORKDIR /app
COPY app.py qwen_adapter.py run.sh ./
RUN chmod +x /app/run.sh

# Portable nginx config base
RUN mkdir -p /opt/ng1/conf /opt/ng1/logs && \
    cp /etc/nginx/mime.types /opt/ng1/conf/mime.types

# Default environment (overrideable at runtime)
ENV APP_DIR=/app \
    OUTPUT_DIR=/app/outputs \
    QUEUE_DIR=/app/queue \
    CSV_INBOX=/app/queue/inbox.csv \
    BASE_URL=http://localhost:8080 \
    MAX_CONCURRENCY=1 \
    DEFAULT_LONG_EDGE=1024 \
    USE_4BIT=1 \
    CSV_SECRET= \
    ADMIN_USER=qwenadmin \
    ADMIN_PASS=changeme \
    NGINX_PREFIX=/opt/ng1 \
    NGINX_PORT=8080 \
    UVICORN_HOST=127.0.0.1 \
    UVICORN_PORT=8000

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/api/healthz || exit 1

CMD ["/bin/bash", "/app/run.sh", "start"]

