"""
Virtual Try-On engine using Replicate API.

No GPU required — inference runs on Replicate's cloud GPUs.
Cost: ~$0.05 per try-on.

Supported models on Replicate:
  - cuuupid/idm-vton (IDM-VTON — best quality)
  - nftnik/virtual-try-on (alternative)
"""

import base64
import logging
import httpx
from io import BytesIO
from PIL import Image

from app.config import REPLICATE_API_TOKEN, REPLICATE_MODEL

logger = logging.getLogger(__name__)

REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"


def image_to_data_uri(image: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 data URI."""
    buffer = BytesIO()
    image.save(buffer, format=format, quality=90)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/jpeg" if format == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


async def run_tryon_replicate(
    person_image: Image.Image,
    garment_image: Image.Image,
    category: str = "upper_body",
    num_steps: int = 30,
    guidance_scale: float = 2.0,
    seed: int = 42,
) -> str:
    """
    Run virtual try-on via Replicate API.

    Returns the URL of the result image.
    """
    if not REPLICATE_API_TOKEN:
        raise ValueError("REPLICATE_API_TOKEN not configured")

    # Resize images to standard VTON input
    person_img = person_image.resize((768, 1024), Image.LANCZOS)
    garment_img = garment_image.resize((768, 1024), Image.LANCZOS)

    # Convert to data URIs
    person_uri = image_to_data_uri(person_img)
    garment_uri = image_to_data_uri(garment_img)

    # Map category to model input
    garment_type_map = {
        "upper_body": "upper_body",
        "lower_body": "lower_body",
        "dresses": "dresses",
    }

    # Build Replicate prediction request (matches cuuupid/idm-vton schema)
    payload = {
        "version": REPLICATE_MODEL,
        "input": {
            "human_img": person_uri,
            "garm_img": garment_uri,
            "garment_des": f"a photo of a {category.replace('_', ' ')} garment",
            "category": garment_type_map.get(category, "upper_body"),
            "steps": num_steps,
            "seed": seed,
            "crop": True,
            "force_dc": category == "dresses",
        },
    }

    headers = {
        "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",  # Wait for result (up to 60s)
    }

    logger.info(f"Submitting try-on to Replicate ({REPLICATE_MODEL})")

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Submit prediction
        resp = await client.post(REPLICATE_API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        prediction = resp.json()

        # If "Prefer: wait" returned completed result
        if prediction.get("status") == "succeeded":
            return _extract_output_url(prediction)

        # Otherwise poll for result
        poll_url = prediction.get("urls", {}).get("get", "")
        if not poll_url:
            poll_url = f"{REPLICATE_API_URL}/{prediction['id']}"

        return await _poll_prediction(client, poll_url, headers)


async def _poll_prediction(client: httpx.AsyncClient, url: str, headers: dict) -> str:
    """Poll Replicate prediction until complete."""
    import asyncio

    for attempt in range(90):  # max ~3 minutes
        await asyncio.sleep(2)

        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        prediction = resp.json()

        status = prediction.get("status")
        logger.debug(f"Prediction status: {status}")

        if status == "succeeded":
            return _extract_output_url(prediction)
        elif status in ("failed", "canceled"):
            error = prediction.get("error", "Unknown error")
            raise RuntimeError(f"Replicate prediction failed: {error}")

    raise TimeoutError("Replicate prediction timed out")


def _extract_output_url(prediction: dict) -> str:
    """Extract output image URL from Replicate prediction response."""
    output = prediction.get("output")
    if isinstance(output, list) and len(output) > 0:
        return output[0]
    elif isinstance(output, str):
        return output
    else:
        raise ValueError(f"Unexpected output format: {output}")
