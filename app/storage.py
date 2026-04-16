import uuid
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from app.config import UPLOAD_DIR, RESULTS_DIR, RESULT_TTL_HOURS, STORAGE_BACKEND, PUBLIC_URL


def save_upload(data: bytes, suffix: str = ".jpg") -> str:
    file_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"
    path = UPLOAD_DIR / file_id
    path.write_bytes(data)
    return str(path)


def save_result(image, job_id: str) -> str:
    filename = f"{job_id}.png"
    path = RESULTS_DIR / filename
    image.save(str(path))
    return filename


def get_result_url(filename: str) -> str:
    if PUBLIC_URL:
        return f"{PUBLIC_URL}/results/{filename}"
    return f"/results/{filename}"


def cleanup_old_results():
    cutoff = datetime.utcnow() - timedelta(hours=RESULT_TTL_HOURS)
    for f in RESULTS_DIR.iterdir():
        if f.is_file() and datetime.utcfromtimestamp(f.stat().st_mtime) < cutoff:
            f.unlink(missing_ok=True)
    for f in UPLOAD_DIR.iterdir():
        if f.is_file() and datetime.utcfromtimestamp(f.stat().st_mtime) < cutoff:
            f.unlink(missing_ok=True)
