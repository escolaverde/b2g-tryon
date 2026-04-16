"""
Virtual Try-On engine using IDM-VTON.

Supports:
  - IDM-VTON (yisol/IDM-VTON) — best quality, requires ~16GB VRAM
  - Falls back to OOTDiffusion or CatVTON if configured

The pipeline:
  1. Person image → pose detection + segmentation (DensePose/OpenPose)
  2. Garment image → mask + features
  3. Diffusion model generates person wearing the garment
"""

import gc
import torch
import logging
from PIL import Image
from pathlib import Path
from typing import Optional

from app.config import VTON_MODEL, DEVICE, DTYPE, MODEL_DIR

logger = logging.getLogger(__name__)

# Global model cache
_pipeline = None
_pose_model = None


def get_torch_dtype():
    if DTYPE == "float16":
        return torch.float16
    return torch.float32


def load_pipeline():
    """Load the IDM-VTON pipeline. Called once at startup."""
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    logger.info(f"Loading VTON model: {VTON_MODEL} on {DEVICE} ({DTYPE})")

    try:
        from diffusers import AutoPipelineForInpainting
        from huggingface_hub import snapshot_download

        # IDM-VTON is based on a custom pipeline
        # We use the official implementation pattern
        model_path = snapshot_download(
            repo_id=VTON_MODEL,
            local_dir=str(MODEL_DIR / VTON_MODEL.replace("/", "_")),
        )

        # Import IDM-VTON specific modules
        import sys
        sys.path.insert(0, model_path)

        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from src.unet_hacked_garmnet import UNet2DConditionModel as GarmentUNet
        from src.unet_hacked_tryon import UNet2DConditionModel as TryonUNet

        # Load base SDXL inpainting
        from diffusers import DDPMScheduler, AutoencoderKL
        from transformers import (
            CLIPImageProcessor,
            CLIPVisionModelWithProjection,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            AutoTokenizer,
        )

        dtype = get_torch_dtype()

        tokenizer_one = AutoTokenizer.from_pretrained(
            model_path, subfolder="tokenizer", use_fast=False
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            model_path, subfolder="tokenizer_2", use_fast=False
        )
        noise_scheduler = DDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        text_encoder_one = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=dtype
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            model_path, subfolder="text_encoder_2", torch_dtype=dtype
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_path, subfolder="image_encoder", torch_dtype=dtype
        )
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=dtype
        )
        unet = TryonUNet.from_pretrained(
            model_path, subfolder="unet", torch_dtype=dtype
        )
        unet_encoder = GarmentUNet.from_pretrained(
            model_path, subfolder="unet_encoder", torch_dtype=dtype
        )

        _pipeline = TryonPipeline.from_pretrained(
            model_path,
            unet=unet,
            vae=vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            scheduler=noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=dtype,
        )

        _pipeline.unet_encoder = unet_encoder
        _pipeline = _pipeline.to(DEVICE)

        logger.info("VTON pipeline loaded successfully")
        return _pipeline

    except Exception as e:
        logger.error(f"Failed to load VTON pipeline: {e}")
        raise


def load_pose_model():
    """Load DensePose / OpenPose model for pose estimation."""
    global _pose_model

    if _pose_model is not None:
        return _pose_model

    try:
        from preprocess.openpose.run_openpose import OpenPose
        from preprocess.humanparsing.run_parsing import Parsing

        _pose_model = {
            "openpose": OpenPose(0 if DEVICE == "cuda" else -1),
            "parsing": Parsing(0 if DEVICE == "cuda" else -1),
        }
        logger.info("Pose models loaded")
        return _pose_model

    except ImportError:
        # Fallback: use a simpler approach with rembg + mediapipe
        logger.warning("OpenPose not available, using fallback pose detection")
        _pose_model = {"fallback": True}
        return _pose_model


def preprocess_person(image: Image.Image) -> dict:
    """Extract pose and segmentation from person image."""
    pose_model = load_pose_model()

    if pose_model.get("fallback"):
        return preprocess_person_fallback(image)

    # Resize to standard VTON input
    image = image.resize((768, 1024), Image.LANCZOS)

    openpose = pose_model["openpose"]
    parsing = pose_model["parsing"]

    keypoints = openpose(image)
    parse_result, _ = parsing(image)

    # Create masks for upper body (garment area)
    import numpy as np
    parse_array = np.array(parse_result)

    # Standard parsing labels: 5=upper, 6=dress, 7=coat
    upper_mask = (parse_array == 5) | (parse_array == 6) | (parse_array == 7)

    mask = Image.fromarray((upper_mask * 255).astype(np.uint8))

    return {
        "image": image,
        "keypoints": keypoints,
        "mask": mask,
        "parse": parse_result,
    }


def preprocess_person_fallback(image: Image.Image) -> dict:
    """Fallback preprocessing without OpenPose — uses rembg + simple masking."""
    import numpy as np

    image = image.resize((768, 1024), Image.LANCZOS)

    # Simple upper body mask using center region heuristic
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    import PIL.ImageDraw as ImageDraw
    draw = ImageDraw.Draw(mask)
    # Upper body region estimate
    draw.rectangle(
        [int(w * 0.15), int(h * 0.1), int(w * 0.85), int(h * 0.65)],
        fill=255,
    )

    return {
        "image": image,
        "keypoints": None,
        "mask": mask,
        "parse": None,
    }


def run_tryon(
    person_image: Image.Image,
    garment_image: Image.Image,
    category: str = "upper_body",
    num_steps: int = 30,
    guidance_scale: float = 2.0,
    seed: int = 42,
) -> Image.Image:
    """
    Run virtual try-on inference.

    Args:
        person_image: Photo of the person
        garment_image: Photo of the garment (flat lay or on mannequin)
        category: upper_body | lower_body | dresses
        num_steps: Diffusion steps (more = better quality, slower)
        guidance_scale: How closely to follow the garment
        seed: Random seed for reproducibility

    Returns:
        PIL Image of the person wearing the garment
    """
    pipeline = load_pipeline()

    # Preprocess
    person_data = preprocess_person(person_image)
    garment_img = garment_image.resize((768, 1024), Image.LANCZOS)

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=get_torch_dtype()):
        result = pipeline(
            image=person_data["image"],
            mask_image=person_data["mask"],
            pose_img=person_data.get("keypoints"),
            cloth=garment_img,
            text="a photo of a person",
            cloth_type=category,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=1024,
            width=768,
        )

    output_image = result.images[0]

    # Cleanup GPU memory
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    return output_image
