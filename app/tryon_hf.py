"""
Virtual Try-On engine using HuggingFace Spaces (Gradio API).

100% FREE — no API key, no credits, no GPU needed locally.
Uses public IDM-VTON Spaces on HuggingFace with automatic fallback.
"""

import logging
import tempfile
import os
import asyncio
from PIL import Image
from gradio_client import Client, handle_file

logger = logging.getLogger(__name__)

# Spaces to try in order (fallback on rate limit)
SPACES = [
    os.getenv("HF_SPACE", "kadirnar/IDM-VTON"),
    "yisol/IDM-VTON",
    "jjlealse/IDM-VTON",
    "LPDoctor/IDM-VTON-demo",
]

_clients = {}


def get_client(space: str) -> Client:
    if space not in _clients:
        logger.info(f"Connecting to HuggingFace Space: {space}")
        _clients[space] = Client(space)
    return _clients[space]


async def run_tryon_hf(
    person_image: Image.Image,
    garment_image: Image.Image,
    category: str = "upper_body",
    num_steps: int = 30,
    guidance_scale: float = 2.0,
    seed: int = 42,
) -> str:
    """
    Run virtual try-on via HuggingFace Space (free).
    Tries multiple spaces with automatic fallback on rate limit.

    Returns the local path of the result image.
    """
    # Save images to temp files (Gradio client needs file paths)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        person_image.save(f, format="JPEG", quality=90)
        person_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        garment_image.save(f, format="JPEG", quality=90)
        garment_path = f.name

    category_desc = {
        "upper_body": "A stylish upper body garment",
        "lower_body": "A stylish lower body garment",
        "dresses": "A stylish dress",
    }

    last_error = None

    try:
        for space in SPACES:
            try:
                logger.info(f"Trying space: {space}")
                client = get_client(space)

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda c=client: c.predict(
                        dict(background=handle_file(person_path), layers=[], composite=None),
                        handle_file(garment_path),
                        category_desc.get(category, "A garment"),
                        True,   # is_checked
                        True,   # is_checked_crop
                        num_steps,
                        seed,
                        api_name="/tryon",
                    )
                )

                if isinstance(result, (list, tuple)):
                    result_path = result[0]
                else:
                    result_path = result

                logger.info(f"Try-on completed via {space}, result at: {result_path}")
                return result_path

            except Exception as e:
                last_error = e
                error_str = str(e)
                logger.warning(f"Space {space} failed: {error_str}")

                # If rate limited or unavailable, try next space
                if "429" in error_str or "503" in error_str or "exceeded" in error_str.lower():
                    # Clear cached client so it reconnects next time
                    _clients.pop(space, None)
                    await asyncio.sleep(2)
                    continue

                # For other errors, also try next space
                _clients.pop(space, None)
                continue

        raise RuntimeError(f"All spaces failed. Last error: {last_error}")

    finally:
        try:
            os.unlink(person_path)
            os.unlink(garment_path)
        except OSError:
            pass
