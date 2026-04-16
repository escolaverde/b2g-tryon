"""
Image preprocessing for better try-on results.

Key improvements:
  1. Remove background from garment images (isolate the clothing)
  2. Ensure proper aspect ratio (3:4 for VTON models)
  3. Validate image quality
"""

import logging
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


def remove_garment_background(image: Image.Image) -> Image.Image:
    """
    Remove background from garment image using rembg.
    This isolates the clothing item, giving MUCH better results
    when the garment photo has a person wearing it or a busy background.
    """
    try:
        from rembg import remove

        logger.info("Removing garment background...")

        # rembg works with bytes
        input_buffer = BytesIO()
        image.save(input_buffer, format="PNG")
        input_bytes = input_buffer.getvalue()

        output_bytes = remove(input_bytes)
        result = Image.open(BytesIO(output_bytes)).convert("RGBA")

        # Put on white background (VTON models prefer white bg)
        white_bg = Image.new("RGBA", result.size, (255, 255, 255, 255))
        white_bg.paste(result, mask=result.split()[3])
        final = white_bg.convert("RGB")

        logger.info("Background removed successfully")
        return final

    except ImportError:
        logger.warning("rembg not installed, skipping background removal")
        return image
    except Exception as e:
        logger.warning(f"Background removal failed: {e}, using original")
        return image


def ensure_aspect_ratio(image: Image.Image, target_ratio: float = 3 / 4) -> Image.Image:
    """
    Ensure image has approximately 3:4 aspect ratio (width:height).
    Pads with white if needed rather than cropping.
    """
    w, h = image.size
    current_ratio = w / h

    if abs(current_ratio - target_ratio) < 0.05:
        return image  # Close enough

    if current_ratio > target_ratio:
        # Too wide — add height
        new_h = int(w / target_ratio)
        padded = Image.new("RGB", (w, new_h), (255, 255, 255))
        offset_y = (new_h - h) // 2
        padded.paste(image, (0, offset_y))
        return padded
    else:
        # Too tall — add width
        new_w = int(h * target_ratio)
        padded = Image.new("RGB", (new_w, h), (255, 255, 255))
        offset_x = (new_w - w) // 2
        padded.paste(image, (offset_x, 0))
        return padded


def preprocess_person(image: Image.Image) -> Image.Image:
    """Preprocess person image for optimal VTON results."""
    image = ensure_aspect_ratio(image)
    image = image.resize((768, 1024), Image.LANCZOS)
    return image


def preprocess_garment(image: Image.Image, remove_bg: bool = True) -> Image.Image:
    """
    Preprocess garment image for optimal VTON results.

    - Removes background (isolates the garment)
    - Ensures 3:4 aspect ratio
    - Resizes to standard VTON input
    """
    if remove_bg:
        image = remove_garment_background(image)
    image = ensure_aspect_ratio(image)
    image = image.resize((768, 1024), Image.LANCZOS)
    return image
