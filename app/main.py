"""
B2G TryOn API — Virtual Try-On as a Service

Endpoints:
  POST /api/tryon        → Submit a try-on job (person + garment images)
  GET  /api/tryon/{id}   → Check job status / get result
  GET  /api/health       → Health check
  GET  /results/{file}   → Serve result images

Supports two inference backends:
  - "replicate" (default) — cloud GPU via Replicate API, no local GPU needed
  - "local" — local GPU with IDM-VTON (requires ≥16GB VRAM)
"""

import uuid
import asyncio
import logging
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image

from app.config import (
    ALLOWED_ORIGINS, API_KEY, MAX_IMAGE_SIZE_MB,
    RESULTS_DIR, UPLOAD_DIR, INFERENCE_BACKEND,
)
from app.storage import save_upload, save_result, get_result_url, cleanup_old_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for local GPU inference
executor = ThreadPoolExecutor(max_workers=2)

# In-memory job store
jobs: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"B2G TryOn API starting (backend={INFERENCE_BACKEND})...")
    cleanup_old_results()
    yield
    executor.shutdown(wait=False)


app = FastAPI(
    title="B2G TryOn API",
    version="1.0.0",
    description="Virtual Try-On API — embed in any e-commerce site",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve result images
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Serve widget JS (if present)
import os
WIDGET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "widget")
if os.path.isdir(WIDGET_DIR):
    app.mount("/widget", StaticFiles(directory=WIDGET_DIR), name="widget")


# --- Auth ---

async def verify_api_key(request: Request):
    if not API_KEY:
        return
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# --- Endpoints ---

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "backend": INFERENCE_BACKEND,
        "gpu": _check_gpu(),
    }


@app.post("/api/tryon", dependencies=[Depends(verify_api_key)])
async def create_tryon(
    person: UploadFile = File(..., description="Photo of the person"),
    garment: UploadFile = File(..., description="Photo of the garment"),
    category: str = Form("upper_body", description="upper_body | lower_body | dresses"),
    num_steps: int = Form(50, ge=10, le=50),
    guidance_scale: float = Form(3.5, ge=1.0, le=5.0),
):
    """Submit a virtual try-on job."""

    # Validate file sizes
    for f, name in [(person, "person"), (garment, "garment")]:
        content = await f.read()
        if len(content) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise HTTPException(400, f"{name} image exceeds {MAX_IMAGE_SIZE_MB}MB limit")
        await f.seek(0)

    # Read images
    try:
        person_bytes = await person.read()
        garment_bytes = await garment.read()
        person_img = Image.open(BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(BytesIO(garment_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image format")

    # Save uploads
    save_upload(person_bytes, ".jpg")
    save_upload(garment_bytes, ".jpg")

    # Create job
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "processing", "result": None, "error": None}

    # Run preprocessing + inference in background
    asyncio.create_task(_run_tryon_async(
        job_id, person_img, garment_img, category, num_steps, guidance_scale,
    ))

    return {
        "job_id": job_id,
        "status": "processing",
        "poll_url": f"/api/tryon/{job_id}",
    }


@app.get("/api/tryon/{job_id}", dependencies=[Depends(verify_api_key)])
async def get_tryon(job_id: str):
    """Check job status and get result."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]

    if job["status"] == "completed":
        return {
            "job_id": job_id,
            "status": "completed",
            "result_url": job["result"],
        }
    elif job["status"] == "error":
        return {
            "job_id": job_id,
            "status": "error",
            "error": job["error"],
        }
    else:
        return {
            "job_id": job_id,
            "status": "processing",
        }


@app.post("/api/tryon/sync", dependencies=[Depends(verify_api_key)])
async def create_tryon_sync(
    person: UploadFile = File(...),
    garment: UploadFile = File(...),
    category: str = Form("upper_body"),
    num_steps: int = Form(50, ge=10, le=50),
    guidance_scale: float = Form(3.5, ge=1.0, le=5.0),
):
    """Synchronous try-on — waits for result."""
    try:
        person_bytes = await person.read()
        garment_bytes = await garment.read()
        person_img = Image.open(BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(BytesIO(garment_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image format")

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "processing", "result": None, "error": None}

    await _run_tryon_async(
        job_id, person_img, garment_img, category, num_steps, guidance_scale,
    )

    job = jobs.get(job_id, {})
    if job.get("status") == "error":
        raise HTTPException(500, job.get("error", "Inference failed"))

    return {
        "job_id": job_id,
        "status": "completed",
        "result_url": job["result"],
    }


# --- Internal ---

async def _run_tryon_async(
    job_id: str,
    person_img: Image.Image,
    garment_img: Image.Image,
    category: str,
    num_steps: int,
    guidance_scale: float,
):
    """Preprocess images and route inference to the configured backend."""
    try:
        # Preprocess in background (rembg can take 10-20s on first run)
        from app.preprocess import preprocess_person, preprocess_garment
        loop = asyncio.get_event_loop()
        person_img = await loop.run_in_executor(executor, preprocess_person, person_img)
        garment_img = await loop.run_in_executor(executor, preprocess_garment, garment_img, True)

        if INFERENCE_BACKEND == "huggingface":
            result_path = await _run_huggingface(
                person_img, garment_img, category, num_steps, guidance_scale,
            )
            # Copy result to our results dir and serve it
            from app.storage import save_result
            from PIL import Image as PILImage
            result_image = PILImage.open(result_path)
            filename = save_result(result_image, job_id)
            result_url = get_result_url(filename)
            jobs[job_id] = {"status": "completed", "result": result_url, "error": None}
        elif INFERENCE_BACKEND == "replicate":
            result_url = await _run_replicate(
                person_img, garment_img, category, num_steps, guidance_scale,
            )
            jobs[job_id] = {"status": "completed", "result": result_url, "error": None}
        else:
            # Local GPU — run in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                executor,
                _run_local_tryon,
                job_id, person_img, garment_img, category, num_steps, guidance_scale,
            )

        logger.info(f"Job {job_id} completed")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id] = {"status": "error", "result": None, "error": str(e)}


async def _run_huggingface(
    person_img, garment_img, category, num_steps, guidance_scale,
) -> str:
    """Run try-on via HuggingFace Space (FREE)."""
    from app.tryon_hf import run_tryon_hf

    return await run_tryon_hf(
        person_image=person_img,
        garment_image=garment_img,
        category=category,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
    )


async def _run_replicate(
    person_img, garment_img, category, num_steps, guidance_scale,
) -> str:
    """Run try-on via Replicate API."""
    from app.tryon_replicate import run_tryon_replicate

    return await run_tryon_replicate(
        person_image=person_img,
        garment_image=garment_img,
        category=category,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
    )


def _run_local_tryon(
    job_id, person_img, garment_img, category, num_steps, guidance_scale,
):
    """Run try-on on local GPU."""
    from app.tryon import run_tryon

    result_image = run_tryon(
        person_image=person_img,
        garment_image=garment_img,
        category=category,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
    )

    filename = save_result(result_image, job_id)
    result_url = get_result_url(filename)
    jobs[job_id] = {"status": "completed", "result": result_url, "error": None}


def _check_gpu() -> dict:
    if INFERENCE_BACKEND == "huggingface":
        return {"backend": "huggingface", "available": True, "cost": "free"}
    if INFERENCE_BACKEND == "replicate":
        return {"backend": "replicate", "available": True}
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "backend": "local",
                "available": True,
                "device": torch.cuda.get_device_name(0),
                "memory_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1),
            }
    except ImportError:
        pass
    return {"backend": "local", "available": False}


if __name__ == "__main__":
    import uvicorn
    from app.config import API_HOST, API_PORT
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=True)
