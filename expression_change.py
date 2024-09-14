import os
import logging
from typing import Optional
from dotenv import load_dotenv
import requests
from PIL import Image
import io
from diffusers import StableDiffusionImg2ImgPipeline
import torch

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hugging Face API constants
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_TOKEN = os.environ.get("HF_TOKEN")

def verify_token(api_token: str) -> bool:
    headers = {"Authorization": f"Bearer {api_token}"}
    try:
        response = requests.get(HF_API_URL, headers=headers)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        logging.error(f"Token verification failed: {str(e)}")
        return False

def change_expression_image2image(image: Image.Image, prompt: str, api_token: str) -> Image.Image:
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_auth_token=api_token)
    pipeline = pipeline.to("cuda")

    output = pipeline(prompt=prompt, image=image, strength=0.75, guidance_scale=7.5).images[0]
    return output

def main(input_image_path: str, output_dir: str, api_token: Optional[str] = None):
    api_token = api_token or HF_TOKEN
    if not api_token:
        raise ValueError("API token is required. Provide it as an argument or set the HF_TOKEN environment variable.")

    if not verify_token(api_token):
        logging.error("Token verification failed. Exiting.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the input image
    input_image = Image.open(input_image_path)

    # Define expressions and their corresponding prompts
    expressions = {
        "Smiling": "a character with a warm, genuine smile",
        "Laughing": "a character laughing heartily",
        "Surprised": "a character with a surprised expression, raised eyebrows",
        "Sad": "a character with a sad, downcast expression",
        "Mad": "a character with an angry, frustrated expression",
        "Afraid": "a character with a fearful, wide-eyed expression"
    }

    # Generate images for each expression using image2image
    for expression, prompt in expressions.items():
        output_image = change_expression_image2image(input_image, prompt, api_token)
        output_path = os.path.join(output_dir, f"{expression.lower()}_modified.png")
        output_image.save(output_path)
        logging.info(f"Generated modified {expression}: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Change expressions of an input image using Stable Diffusion")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_dir", help="Directory to save output images")
    parser.add_argument("--api_token", help="Hugging Face API token")
    args = parser.parse_args()

    main(args.input_image, args.output_dir, args.api_token)
