import os
import csv
import asyncio
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from qwen_adapter import QwenImageEdit

# --------------------
# Configuration
# --------------------
MODEL_ID = os.environ.get("MODEL_ID", "/opt/models/Qwen-Image-Edit")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/outputs"))
QUEUE_DIR = Path(os.environ.get("QUEUE_DIR", "/app/queue"))
CSV_INBOX = Path(os.environ.get("CSV_INBOX", str(QUEUE_DIR / "inbox.csv")))
CSV_SECRET = os.environ.get("CSV_SECRET", "")
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "1"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_DIR.mkdir(parents=True, exist_ok=True)
CSV_INBOX.parent.mkdir(parents=True, exist_ok=True)
if not CSV_INBOX.exists():
    CSV_INBOX.write_text("image_url,prompt,directory\n")

# --------------------
# Init FastAPI
# --------------------
app = FastAPI(title="Qwen Image Edit API")

qwen: Optional[QwenImageEdit] = None
_worker_task: Optional[asyncio.Task] = None
_csv_lock = asyncio.Lock()


# --------------------
# Startup / Shutdown
# --------------------
@app.on_event("startup")
def _startup():
    global qwen, _worker_task
    qwen = QwenImageEdit(backend="local", model_id=MODEL_ID, device="cuda")
    _worker_task = asyncio.create_task(_csv_worker())


@app.on_event("shutdown")
def _shutdown():
    global qwen, _worker_task
    if _worker_task:
        _worker_task.cancel()


# --------------------
# Healthz
# --------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# --------------------
# One-off edit
# --------------------
@app.post("/api/edit")
async def edit_image(
    prompt: str = Form(...),
    image_url: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
    directory: str = Form("manual"),
):
    if not qwen:
        raise HTTPException(status_code=500, detail="Model not initialized")

    # load image
    if image_url:
        resp = requests.get(image_url, stream=True)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch image_url")
        data = resp.content
        filename = Path(image_url).name
    elif image_file:
        data = await image_file.read()
        filename = image_file.filename
    else:
        raise HTTPException(status_code=400, detail="Either image_url or image_file is required")

    outdir = OUTPUT_DIR / "manual" / directory
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = await qwen.edit_async(prompt=prompt, image_bytes=data, outdir=outdir)
    return FileResponse(outfile, media_type="image/png", filename=outfile.name)


# --------------------
# CSV inbox endpoints
# --------------------
def _check_secret(secret: Optional[str]):
    if CSV_SECRET and secret != CSV_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")


@app.get("/csv")
def get_csv(secret: Optional[str] = None):
    _check_secret(secret)
    return PlainTextResponse(CSV_INBOX.read_text())


@app.post("/csv/append")
async def append_csv(lines: str, secret: Optional[str] = None):
    _check_secret(secret)
    async with _csv_lock:
        with CSV_INBOX.open("a") as f:
            if not lines.endswith("\n"):
                lines += "\n"
            f.write(lines)
    return {"status": "ok"}


@app.get("/csv/ui")
def csv_ui(secret: Optional[str] = None):
    _check_secret(secret)
    rows = CSV_INBOX.read_text().strip().splitlines()
    if not rows:
        return PlainTextResponse("Inbox empty")

    html = ["<html><body><h2>CSV Inbox</h2><table border='1'>"]
    for row in csv.reader(rows):
        html.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    html.append("</table></body></html>")
    return PlainTextResponse("\n".join(html), media_type="text/html")


# --------------------
# Background CSV worker
# --------------------
async def _csv_worker():
    print("[csv_worker] started")
    while True:
        try:
            await asyncio.sleep(5)
            async with _csv_lock:
                lines = CSV_INBOX.read_text().strip().splitlines()
                if len(lines) <= 1:
                    continue  # header only or empty
                header, *rows = lines

                remaining = [header]
                for row in rows:
                    try:
                        image_url, prompt, directory = row.split(",", 2)
                    except Exception:
                        continue
                    try:
                        resp = requests.get(image_url, stream=True, timeout=20)
                        if resp.status_code != 200:
                            raise RuntimeError(f"fetch failed {resp.status_code}")
                        data = resp.content
                        outdir = OUTPUT_DIR / directory
                        outdir.mkdir(parents=True, exist_ok=True)
                        await qwen.edit_async(prompt=prompt, image_bytes=data, outdir=outdir)
                    except Exception as e:
                        print("[csv_worker] failed:", row, e)
                        remaining.append(row)

                # overwrite inbox with any remaining
                CSV_INBOX.write_text("\n".join(remaining) + "\n")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print("[csv_worker] error:", e)
