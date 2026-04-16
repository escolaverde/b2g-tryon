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
    os.getenv("HF_SPACE", "yisol/IDM-VTON"),
    "ronniechoyy/IDM-VTON-20250428",
    "jjlealse/IDM-VTON",
]

_clients = {}


def _try_predict(client, person_path, garment_path, garment_desc, num_steps, seed):
    """Try different input formats for compatibility across spaces."""
    person_file = handle_file(person_path)
    garment_file = handle_file(garment_path)

    # Format 1: ImageEditor dict (newer spaces)
    try:
        return client.predict(
            dict(background=person_file, layers=[], composite=None),
            garment_file,
            garment_desc,
            True, True, num_steps, seed,
            api_name="/tryon",
        )
    except (AttributeError, TypeError):
        pass

    # Format 2: Simple file handle (older spaces)
    try:
        return client.predict(
            person_file,
            garment_file,
            garment_desc,
            True, True, num_steps, seed,
            api_name="/tryon",
        )
    except (AttributeError, TypeError):
        pass

    # Format 3: With explicit image editor format
    return client.predict(
        {"background": person_file, "layers": None, "composite": None},
        garment_file,
        garment_desc,
        True, True, num_steps, seed,
        api_name="/tryon",
    )


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
                    lambda c=client: _try_predict(c, person_path, garment_path,
                        category_desc.get(category, "A garment"), num_steps, seed),
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

                # Clear cached client so it reconnects next time
                _clients.pop(space, None)

                # If rate limited, paused, or unavailable, try next space
                recoverable = ["429", "503", "exceeded", "paused", "invalid state", "runtime_error"]
                if any(keyword in error_str.lower() for keyword in recoverable):
                    await asyncio.sleep(2)
                    continue

                # For other errors, also try next space
                continue

        raise RuntimeError(f"All spaces failed. Last error: {last_error}")

    finally:
        try:
            os.unlink(person_path)
            os.unlink(garment_path)
        except OSError:
            pass
