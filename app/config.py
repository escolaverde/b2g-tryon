import os
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# --- API ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_KEY = os.getenv("TRYON_API_KEY", "")  # empty = no auth (dev mode)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
RESULT_TTL_HOURS = int(os.getenv("RESULT_TTL_HOURS", "24"))

# --- Inference Backend ---
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "replicate")  # replicate | local

# Replicate (cloud GPU — no local GPU needed)
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
REPLICATE_MODEL = os.getenv(
    "REPLICATE_MODEL",
    "0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985",
)

# Local GPU (requires NVIDIA GPU with ≥16GB VRAM)
VTON_MODEL = os.getenv("VTON_MODEL", "yisol/IDM-VTON")
DEVICE = os.getenv("DEVICE", "cuda")  # cuda | cpu
DTYPE = os.getenv("DTYPE", "float16")  # float16 | float32

# --- Storage ---
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")  # local | s3
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_PREFIX = os.getenv("S3_PREFIX", "tryon/")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")  # e.g. https://tryon.b2g.com.br
