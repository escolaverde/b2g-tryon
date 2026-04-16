"""
Virtual Try-On engine using HuggingFace Spaces (Gradio API).

100% FREE — no API key, no credits, no GPU needed locally.
Uses the public IDM-VTON Space on HuggingFace.
"""

import logging
import tempfile
import os
from PIL import Image
from gradio_client import Client, handle_file

logger = logging.getLogger(__name__)

# Public HuggingFace Space with IDM-VTON
HF_SPACE = os.getenv("HF_SPACE", "yisol/IDM-VTON")

_client = None


def get_client():
    global _client
    if _client is None:
        logger.info(f"Connecting to HuggingFace Space: {HF_SPACE}")
        _client = Client(HF_SPACE)
    return _client


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

    Returns the local path of the result image.
    """
    import asyncio

    # Save images to temp files (Gradio client needs file paths)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        person_image.save(f, format="JPEG", quality=90)
        person_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        garment_image.save(f, format="JPEG", quality=90)
        garment_path = f.name

    try:
        # Map category
        is_checked_crop = True  # auto-crop for non 3:4 images

        # Category descriptions
        category_desc = {
            "upper_body": "A stylish upper body garment",
            "lower_body": "A stylish lower body garment",
            "dresses": "A stylish dress",
        }

        client = get_client()

        # Run prediction in thread pool (gradio_client is synchronous)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: client.predict(
                dict(background=handle_file(person_path), layers=[], composite=None),  # person image
                handle_file(garment_path),  # garment image
                category_desc.get(category, "A garment"),  # garment description
                True,   # is_checked (use auto-generated mask)
                True,   # is_checked_crop (auto crop)
                num_steps,  # denoise steps
                seed,   # seed
                api_name="/tryon",
            )
        )

        # Result is a tuple — the output image path is the first element
        if isinstance(result, (list, tuple)):
            result_path = result[0]
        else:
            result_path = result

        logger.info(f"Try-on completed, result at: {result_path}")
        return result_path

    finally:
        # Cleanup temp files
        try:
            os.unlink(person_path)
            os.unlink(garment_path)
        except OSError:
            pass
