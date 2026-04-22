"""React App — FastAPI backend that serves both API and React static files.

Single port deployment: FastAPI serves the React build AND exposes /api/* endpoints.
"""

import asyncio
import gc
import io
import os
import queue
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from urllib.parse import quote

import torch

# Add parent dirs to path so task1/task2/task3 packages are importable
_APP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np

from model_manager import manager
from task1 import preload_to_cpu as preload_task1_cpu
from task2 import preload_to_cpu as preload_task2_cpu
from task3 import preload_to_cpu as preload_task3_cpu



# ══ GPU inference queue ════════════════════════════════════════════════════════════════
# Single worker serialises ALL GPU jobs across tasks.
# Multiple users get job_id immediately and poll for result.
_GPU_QUEUE: queue.Queue = queue.Queue()


def _gpu_worker():
    while True:
        job_fn, job_id = _GPU_QUEUE.get()
        try:
            job_fn()
        except Exception as e:
            print(f"[GPUWorker] Unhandled error for job {job_id}: {e}")
        finally:
            _GPU_QUEUE.task_done()


threading.Thread(target=_gpu_worker, daemon=True, name="gpu-worker").start()


# ══ Lifespan: startup / shutdown ══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Deferred CUDA init — runs AFTER container env is fully set up by NVIDIA runtime
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.95, device=0)
        except Exception as e:
            print(f"[Startup] CUDA memory fraction warning (non-fatal): {e}")

    print("[Startup] Preloading Task 1 weights to CPU RAM…")
    preload_task1_cpu()
    print("[Startup] Preloading Task 2 pipeline to CPU RAM…")
    preload_task2_cpu()
    print("[Startup] Preloading Task 3 pipeline to CPU RAM…")
    preload_task3_cpu()

    await asyncio.sleep(2)  # let background preload threads get going

    print("[Startup] Activating Task 1 on GPU (default)…")
    try:
        manager.activate("task1")
        print("[Startup] Ready.")
    except Exception as e:
        print(f"[Startup] WARNING: Could not pre-activate task1 on GPU: {e}")
        print("[Startup] App continues — GPU will be activated on first request.")

    yield
    print("[Shutdown] App shutting down.")


# ── FastAPI app ───────────────────────────────────────────────

app = FastAPI(title="Image Processing Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Info"],
)

# ── Async Job Queue ────────────────────────────────────────────────────────────
# Stores job state: {job_id: {"status": "pending|running|done|error", "result": bytes|None, "info": str, "error": str}}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

def _new_job() -> str:
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "result": None, "info": "", "error": ""}
    return job_id

def _set_job_running(job_id: str):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "running"

def _set_job_done(job_id: str, result_bytes: bytes, info: str = ""):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update({"status": "done", "result": result_bytes, "info": info})

def _set_job_error(job_id: str, error: str):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update({"status": "error", "error": error})


def _pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=95)
    buf.seek(0)
    return buf.getvalue()


def _read_upload_image(upload: UploadFile) -> Image.Image:
    return Image.open(io.BytesIO(upload.file.read())).convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
# Task 1 — Multi-Exposure Fusion
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/task1/run")
async def task1_run(
    files: list[UploadFile] = File(...),
    apply_phase2: bool = Form(True),
    align: bool = Form(True),
    brightness: float = Form(1.0),
):
    """Multi-exposure fusion: queues GPU job, returns job_id immediately."""
    import tempfile
    from task1.inference import run as task1_run_fn

    if not files:
        raise HTTPException(400, "No files uploaded")

    # Read file contents eagerly while the request is still open
    file_contents = []
    for f in files:
        content = await f.read()
        file_contents.append((f.filename or "img.png", content))

    job_id = _new_job()

    def _run():
        temp_paths: list[str] = []
        try:
            _set_job_running(job_id)
            for filename, content in file_contents:
                suffix = os.path.splitext(filename)[1] or ".png"
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                tmp.write(content)
                tmp.flush()
                tmp.close()
                temp_paths.append(tmp.name)
            img, info_text = task1_run_fn(temp_paths, apply_phase2, align, brightness=brightness)
            _set_job_done(job_id, _pil_to_bytes(img), info_text)
        except Exception as e:
            import traceback
            _set_job_error(job_id, str(e))
            print(f"[task1/run] job {job_id} error: {e}")
            traceback.print_exc()
        finally:
            for p in temp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
            gc.collect()

    _GPU_QUEUE.put((_run, job_id))
    return {"job_id": job_id}


# ══════════════════════════════════════════════════════════════════════════════
# Task 2 — Inpainting & Editing
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/task2/run")
async def task2_run(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    task_type: str = Form("Add"),
    steps: int = Form(20),
    strength: float = Form(1.0),
    guidance_scale: float = Form(12.0),
    cn_scale: float = Form(0.3),
):
    """Inpainting: accepts image + mask, starts async job, returns job_id immediately."""
    from task2.inference import run_inference

    img_bytes_raw = await image.read()
    mask_bytes_raw = await mask.read()

    job_id = _new_job()

    def _run():
        try:
            _set_job_running(job_id)
            img_pil = Image.open(io.BytesIO(img_bytes_raw)).convert("RGB")
            mask_pil = Image.open(io.BytesIO(mask_bytes_raw)).convert("L")

            mask_arr = np.array(mask_pil)
            rgba_arr = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
            rgba_arr[mask_arr > 127, 3] = 255
            rgba_arr[mask_arr > 127, 0] = 255
            mask_layer = Image.fromarray(rgba_arr, "RGBA")

            editor_val = {
                "background": img_pil,
                "layers": [mask_layer],
                "composite": img_pil,
            }

            result_img, info_text = run_inference(
                editor_val, prompt, steps, task_type,
                strength, guidance_scale, cn_scale,
                neg_prompt_override=negative_prompt,
            )
            result_bytes = _pil_to_bytes(result_img)
            _set_job_done(job_id, result_bytes, info_text)
        except Exception as e:
            import traceback
            _set_job_error(job_id, str(e))
            print(f"[task2/run] job {job_id} error: {e}")
            traceback.print_exc()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _GPU_QUEUE.put((_run, job_id))
    return {"job_id": job_id}


@app.post("/api/task2/enhance-prompt")
async def task2_enhance_prompt(prompt: str = Form(...)):
    """Enhance prompt using Qwen3. Queued through GPU queue to avoid race conditions."""
    import json as _json
    from task2.inference import enhance_prompt as _qwen_enhance

    if not prompt.strip():
        raise HTTPException(400, "Empty prompt")

    job_id = _new_job()

    def _run():
        try:
            _set_job_running(job_id)
            pos, neg = _qwen_enhance(prompt)
            _set_job_done(job_id, b"", _json.dumps({"positive": pos, "negative": neg}))
        except Exception as e:
            import traceback
            _set_job_error(job_id, str(e))
            traceback.print_exc()

    _GPU_QUEUE.put((_run, job_id))
    return {"job_id": job_id}


# ══════════════════════════════════════════════════════════════════════════════
# Task 3 — Outpainting
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/task3/preview")
async def task3_preview(
    image: UploadFile = File(...),
    target_res: str = Form("1:1"),
    custom_w: int = Form(1024),
    custom_h: int = Form(1024),
    alignment: str = Form("Middle"),
    resize_option: str = Form("Full"),
    custom_resize_pct: int = Form(100),
    overlap_percentage: int = Form(10),
    overlap_left: int = Form(10),
    overlap_right: int = Form(10),
    overlap_top: int = Form(10),
    overlap_bottom: int = Form(10),
    pad_left: int = Form(0),
    pad_right: int = Form(0),
    pad_top: int = Form(0),
    pad_bottom: int = Form(0),
):
    from task3.inference import preview as task3_preview_fn

    img_pil = _read_upload_image(image)
    result = task3_preview_fn(
        img_pil, target_res, custom_w, custom_h,
        alignment, resize_option, custom_resize_pct,
        overlap_percentage, overlap_left, overlap_right,
        overlap_top, overlap_bottom,
        pad_left, pad_right, pad_top, pad_bottom,
    )
    if result is None:
        raise HTTPException(400, "Cannot generate preview")

    img_bytes = _pil_to_bytes(result)
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")


@app.post("/api/task3/run")
async def task3_run(
    image: UploadFile = File(...),
    target_res: str = Form("1:1"),
    custom_w: int = Form(1024),
    custom_h: int = Form(1024),
    alignment: str = Form("Middle"),
    resize_option: str = Form("Full"),
    custom_resize_pct: int = Form(100),
    overlap_percentage: int = Form(10),
    overlap_left: int = Form(10),
    overlap_right: int = Form(10),
    overlap_top: int = Form(10),
    overlap_bottom: int = Form(10),
    pad_left: int = Form(0),
    pad_right: int = Form(0),
    pad_top: int = Form(0),
    pad_bottom: int = Form(0),
    prompt: str = Form(""),
    num_steps: int = Form(8),
    sharpen: float = Form(1.0),
    lora_scale: float = Form(0.0),
):
    """Outpainting: starts async job, returns job_id immediately."""
    from task3.inference import infer as task3_infer_fn

    img_bytes_raw = await image.read()
    job_id = _new_job()

    def _run():
        try:
            _set_job_running(job_id)
            img_pil = Image.open(io.BytesIO(img_bytes_raw)).convert("RGB")

            last_result = None
            for _cnet_img, out_img in task3_infer_fn(
                img_pil, target_res, custom_w, custom_h,
                alignment, resize_option, custom_resize_pct,
                overlap_percentage, overlap_left, overlap_right,
                overlap_top, overlap_bottom,
                pad_left, pad_right, pad_top, pad_bottom,
                prompt, num_steps, sharpen, lora_scale,
            ):
                last_result = out_img

            if last_result is None:
                _set_job_error(job_id, "Inference produced no output")
                return

            result_bytes = _pil_to_bytes(last_result)
            _set_job_done(job_id, result_bytes, "")
        except Exception as e:
            import traceback
            _set_job_error(job_id, str(e))
            print(f"[task3/run] job {job_id} error: {e}")
            traceback.print_exc()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _GPU_QUEUE.put((_run, job_id))
    return {"job_id": job_id}


# ══════════════════════════════════════════════════════════════════════════════
# Job status & result endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/job/{job_id}/status")
async def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    resp = {"status": job["status"], "error": job["error"]}
    # Text-only jobs (enhance-prompt): return text in status and clean up immediately
    if job["status"] == "done" and job.get("result") == b"":
        resp["text"] = job.get("info", "")
        with _jobs_lock:
            _jobs.pop(job_id, None)
    return resp


@app.get("/api/job/{job_id}/result")
async def job_result(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(400, f"Job not done yet: {job['status']}")
    result_bytes = job["result"]
    info = job.get("info", "")
    # Clean up after serving
    with _jobs_lock:
        _jobs.pop(job_id, None)
    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="image/png",
        headers={"X-Info": quote(info, safe='')},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tab switch — activate model on GPU
# ══════════════════════════════════════════════════════════════════════════════

_activate_seq = 0          # monotonic counter: mỗi request tăng 1
_activate_seq_lock = threading.Lock()


@app.post("/api/activate/{task_name}")
async def activate_task(task_name: str):
    global _activate_seq
    valid = {"task1", "task2", "task3"}
    if task_name not in valid:
        raise HTTPException(400, f"Invalid task: {task_name}")

    with _activate_seq_lock:
        _activate_seq += 1
        my_seq = _activate_seq          # snapshot cho thread này

    def _do():
        # Trước khi bắt đầu activate (có thể mất 30-60s), kiểm tra xem có request
        # mới hơn không — nếu có thì bỏ qua luôn, không cần chuyển model
        with _activate_seq_lock:
            if my_seq != _activate_seq:
                print(f"[activate] Skipping stale request '{task_name}' (seq {my_seq}, latest {_activate_seq})")
                return
        manager.activate(task_name)

    threading.Thread(target=_do, daemon=True).start()
    return {"status": "activating", "task": task_name}


# ══════════════════════════════════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "active_task": manager.active_task,
        "gpu_available": torch.cuda.is_available(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Example / Demo files
# ══════════════════════════════════════════════════════════════════════════════

_EXAMPLES_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

@app.get("/api/examples/{task}/{rest:path}")
async def serve_example(task: str, rest: str):
    """Serve example files from task{N}/example/... directories."""
    if task not in ("task1", "task2", "task3"):
        raise HTTPException(400, "Invalid task")
    safe = os.path.normpath(rest)
    if safe.startswith("..") or safe.startswith("/"):
        raise HTTPException(400, "Invalid path")
    fpath = os.path.join(_EXAMPLES_ROOT, task, "example", safe)
    if not os.path.isfile(fpath):
        raise HTTPException(404, "File not found")
    return FileResponse(fpath)


# ══════════════════════════════════════════════════════════════════════════════
# Serve React static files (MUST be last — catch-all)
# ══════════════════════════════════════════════════════════════════════════════

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist")

if os.path.isdir(_FRONTEND_DIR):
    # Serve static assets
    _ASSETS_DIR = os.path.join(_FRONTEND_DIR, "assets")
    if os.path.isdir(_ASSETS_DIR):
        app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        # Try to serve as static file first
        file_path = os.path.join(_FRONTEND_DIR, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        # Fallback to index.html (SPA routing)
        return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))
else:
    @app.get("/")
    async def no_frontend():
        return {"message": "Frontend not built yet. Run: cd react_app/frontend && npm run build"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
