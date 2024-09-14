import os
import requests
from PIL import Image
from io import BytesIO
import logging
from diffusers import StableDiffusionImg2ImgPipeline
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CONFIG = {
    "HF_MODEL": "stabilityai/stable-diffusion-xl-base-1.0",
    "STEPS": 30,
    "GUIDANCE_SCALE": 7.5,
    "STRENGTH": 0.7,
    "SCHEDULER": "normal",
    "SEED": 152,
    "IMAGE_CFG_SCALE": 1.5,
    "PROMPT_STRENGTH": 0.8,
}

# Hugging Face API constants
HF_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")

if not HF_API_TOKEN:
    logging.error("HUGGINGFACE_API_TOKEN environment variable is not set. Please set it with your API token.")
    raise ValueError("API token is required.")

def load_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return Image.open(image_path).convert("RGB")
    except (IOError, OSError, Image.DecompressionBombError) as e:
        logging.error(f"Error loading image {image_path}: {str(e)}")
        return None

def change_expression(input_image, expression, device="cuda" if torch.cuda.is_available() else "cpu"):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(CONFIG["HF_MODEL"], torch_dtype=torch.float16, use_auth_token=HF_API_TOKEN)
    pipe = pipe.to(device)

    prompt = f"portrait of a 2D character, {expression} expression, consistent features, detailed, high quality"
    negative_prompt = "low quality, blurry, distorted features, inconsistent style"

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        num_inference_steps=CONFIG["STEPS"],
        guidance_scale=CONFIG["GUIDANCE_SCALE"],
        strength=CONFIG["STRENGTH"],
        seed=CONFIG["SEED"]
    ).images[0]

    return image

def process_image(input_image_path, expression, output_dir):
    input_image = load_image(input_image_path)
    if input_image is None:
        logging.error(f"Failed to load input image: {input_image_path}")
        return None

    try:
        output_image = change_expression(input_image, expression)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"{expression.lower()}_output.png")
        output_image.save(output_filename)
        logging.info(f"Generated and saved {expression} expression: {output_filename}")
        return output_filename
    except Exception as e:
        logging.error(f"Error processing {expression} expression: {str(e)}")
        return None

if __name__ == "__main__":
    # This section can be used for testing the module
    input_image_path = "path/to/your/input/image.png"
    output_dir = "output"
    expression = "smiling"
    result = process_image(input_image_path, expression, output_dir)
    if result:
        print(f"Successfully generated {expression} expression: {result}")
    else:
        print(f"Failed to generate {expression} expression")
