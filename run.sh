#!/usr/bin/env bash
set -e

# ------------ runtime config (overridable via env) ------------
APP_DIR="${APP_DIR:-/app}"

# Ensure base directory exists
mkdir -p /workspace/models

# Clone only if repo directory does not exist
if [ ! -d "/workspace/models/Qwen-Image-Edit/.git" ]; then
    git clone https://huggingface.co/Qwen/Qwen-Image-Edit /workspace/models/Qwen-Image-Edit
fi

# Move into repo dir
cd /workspace/models/Qwen-Image-Edit

# Make sure LFS is available; only pull if needed
if command -v git-lfs &> /dev/null; then
    git lfs pull
else
    echo "Warning: git-lfs not installed, skipping LFS pull"
fi

# Export MODEL_ID (idempotent)
export MODEL_ID="/workspace/models/Qwen-Image-Edit"
MODEL_ID="${MODEL_ID:-/workspace/models/Qwen-Image-Edit}"

# app data (kept inside the container; you can mount if you want persistence)
OUTPUT_DIR="${OUTPUT_DIR:-/app/outputs}"
QUEUE_DIR="${QUEUE_DIR:-/app/queue}"
CSV_INBOX="${CSV_INBOX:-$QUEUE_DIR/inbox.csv}"

# app behavior
BASE_URL="${BASE_URL:-http://localhost:8080}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
DEFAULT_LONG_EDGE="${DEFAULT_LONG_EDGE:-1024}"
USE_4BIT="${USE_4BIT:-1}"
CSV_SECRET="${CSV_SECRET:-}"

# portable nginx
NGINX_PREFIX="${NGINX_PREFIX:-/opt/ng1}"
NGINX_CONF="$NGINX_PREFIX/conf/nginx.conf"
NGINX_LOG_DIR="$NGINX_PREFIX/logs"
NGINX_HTPASS="$NGINX_PREFIX/conf/.htpasswd"
NGINX_PORT="${NGINX_PORT:-8080}"

# auth for /files
ADMIN_USER="${ADMIN_USER:-qwenadmin}"
ADMIN_PASS="${ADMIN_PASS:-changeme}"

# uvicorn endpoint
UVICORN_HOST="${UVICORN_HOST:-127.0.0.1}"
UVICORN_PORT="${UVICORN_PORT:-8000}"

# transformers tip to avoid importing torchvision
export TRANSFORMERS_NO_TORCHVISION="${TRANSFORMERS_NO_TORCHVISION:-1}"

log(){ echo "[run.sh] $*"; }

ensure_dirs() {
  mkdir -p "$OUTPUT_DIR" "$QUEUE_DIR"/{incoming,processing,done,failed,logs}
  [[ -f "$CSV_INBOX" ]] || echo "image_url,prompt,directory" > "$CSV_INBOX"

  # allow nginx to traverse (portable runs as root, but keep permissive)
  chmod a+rx /app || true
  chmod -R a+rx "$OUTPUT_DIR" || true
}

ensure_htpasswd() {
  mkdir -p "$(dirname "$NGINX_HTPASS")"
  if [[ ! -f "$NGINX_HTPASS" ]]; then
    if command -v htpasswd >/dev/null 2>&1; then
      htpasswd -bc "$NGINX_HTPASS" "$ADMIN_USER" "$ADMIN_PASS"
    else
      printf "%s:%s\n" "$ADMIN_USER" "$(openssl passwd -apr1 "$ADMIN_PASS")" > "$NGINX_HTPASS"
    fi
  fi
}

write_nginx_conf() {
  mkdir -p "$NGINX_PREFIX/conf" "$NGINX_LOG_DIR"
  [[ -f "$NGINX_PREFIX/conf/mime.types" ]] || cp /etc/nginx/mime.types "$NGINX_PREFIX/conf/mime.types" || true

  cat > "$NGINX_CONF" <<NG
user root;
worker_processes 1;

events { worker_connections 1024; }

http {
    include       $NGINX_PREFIX/conf/mime.types;
    default_type  application/octet-stream;
    client_max_body_size 50m;
    sendfile on;

    access_log $NGINX_LOG_DIR/access.log;
    error_log  $NGINX_LOG_DIR/error.log notice;

    server {
        listen $NGINX_PORT;
        server_name _;

        # Browse saved images (basic auth)
        location /files/ {
            alias $OUTPUT_DIR/;
            autoindex on;
            auth_basic "Restricted";
            auth_basic_user_file $NGINX_HTPASS;
        }

        # /api/* -> backend /*  (strip /api by using trailing slash)
        location /api/ {
            proxy_pass http://$UVICORN_HOST:$UVICORN_PORT/;
            proxy_http_version 1.1;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_read_timeout 600;
        }

        # exact /csv -> backend /csv
        location = /csv {
            proxy_pass http://$UVICORN_HOST:$UVICORN_PORT/csv;
            proxy_http_version 1.1;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_read_timeout 600;
        }

        # /csv/... -> preserve path (NO trailing slash in proxy_pass)
        location ^~ /csv/ {
            proxy_pass http://$UVICORN_HOST:$UVICORN_PORT;
            proxy_http_version 1.1;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_read_timeout 600;
        }
    }
}
NG
}

start_uvicorn() {
  # stop any leftovers
  pkill -f 'uvicorn app:app' 2>/dev/null || true
  cd "$APP_DIR"
  # app env
  export MODEL_ID OUTPUT_DIR QUEUE_DIR CSV_INBOX BASE_URL \
         MAX_CONCURRENCY DEFAULT_LONG_EDGE USE_4BIT CSV_SECRET
  log "starting uvicorn on ${UVICORN_HOST}:${UVICORN_PORT}…"
  nohup uvicorn app:app --host "$UVICORN_HOST" --port "$UVICORN_PORT" --log-level info \
    > /tmp/uvicorn.out 2>&1 &
}

start_nginx_fg() {
  # free port if something stale is bound
  local holders
  holders=$(ss -ltnp | awk -v p=":${NGINX_PORT}$" '$4 ~ p {print $7}' | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u || true)
  for p in $holders; do kill -TERM "$p" 2>/dev/null || true; done
  rm -f /run/nginx.pid || true

  write_nginx_conf
  ensure_htpasswd

  log "starting portable nginx on :$NGINX_PORT (foreground)…"
  nginx -p "$NGINX_PREFIX" -c "conf/nginx.conf" -t
  exec nginx -g "daemon off;" -p "$NGINX_PREFIX" -c "conf/nginx.conf"
}

case "${1:-start}" in
  start)
    ensure_dirs
    start_uvicorn
    start_nginx_fg
    ;;
  *)
    echo "Usage: $0 start"
    exit 1
    ;;
esac



