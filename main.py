import argparse
import base64
import io
from io import BytesIO
from PIL import Image, ImageDraw
import requests
import json
import logging
import os
import sys
import time
from typing import Optional
from dotenv import load_dotenv
from diffusers import StableDiffusionImg2ImgPipeline
import torch
import random

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Output folder for generated expressions
OUTPUT_FOLDER = "generated_expressions"

# Configuration
CONFIG = {
    "HF_MODEL": "stabilityai/stable-diffusion-xl-base-1.0",
    "VAE_MODEL": "stabilityai/sdxl-vae",
    "LORA_MODEL": "pytorch_lora_weights_SD.safetensors",
    "STEPS": 30,
    "GUIDANCE_SCALE": 7.5,
    "STRENGTH": 0.7,
    "SAMPLER": "lcm",
    "SCHEDULER": "normal",
    "LORA_STRENGTH": 1.0,
    "DENOISE": 0.75,
    "SEED": 152,
    "CLIP_SKIP": 1,
    "VAE_TILING": False,
    "NOISE_MASK_FEATHER": 0,
}

# Hugging Face API constants
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_TOKEN = os.environ.get("HF_TOKEN")
MOCK_API_TOKEN = "mock_token"

def verify_token(api_token: str) -> bool:
    if api_token == MOCK_API_TOKEN:
        return True
    headers = {"Authorization": f"Bearer {api_token}"}
    try:
        response = requests.get(HF_API_URL, headers=headers)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        logging.error(f"Token verification failed: {str(e)}")
        return False

if not HF_TOKEN:
    logging.error("HUGGINGFACE_API_TOKEN environment variable is not set. Please set it with your API token.")
    sys.exit(1)

# Headers for Hugging Face API requests
HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def load_image(image_path):
    try:
        with Image.open(image_path) as img:
            # Verify that the image can be opened
            img.verify()
        # Open the image again and return it as a PIL Image object
        return Image.open(image_path)
    except (IOError, OSError, Image.DecompressionBombError) as e:
        logging.error(f"Error loading image {image_path}: {str(e)}")
        return None

# The get_available_models function is removed as it's not needed for the Hugging Face API.
# We're using a specific model (Stable Diffusion XL) directly in this script.

def generate_expression(
    api_url: str,
    headers: dict,
    prompt: str
) -> Image.Image:

    # Prepare the payload
    payload = {
        "inputs": prompt
    }

    # Update headers
    headers.update({"Content-Type": "application/json"})

    try:
        # Print the full request payload for debugging
        logging.info(f"Request Payload: {payload}")

        # Send the request with JSON payload
        response = requests.post(
            api_url,
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        # Process the response
        content_type = response.headers.get('content-type', '')
        if content_type.startswith('image/'):
            output_image = Image.open(io.BytesIO(response.content))
            return output_image
        elif content_type == 'application/json':
            response_data = response.json()
            if isinstance(response_data, list) and len(response_data) > 0:
                # Handle list response
                image_data = base64.b64decode(response_data[0])
                return Image.open(io.BytesIO(image_data))
            elif isinstance(response_data, dict):
                # Handle dictionary response
                error_msg = response_data.get('error', 'Unknown error occurred')
                raise requests.RequestException(f"API Error: {error_msg}")
            else:
                raise requests.RequestException("Unexpected JSON response format")
        else:
            raise requests.RequestException(f"Unexpected content type: {content_type}")
    except requests.RequestException as e:
        logging.error(f"Error in generate_expression: {str(e)}")
        logging.error(f"Request URL: {api_url}")
        logging.error(f"Request Headers: {headers}")
        logging.error(f"Request Prompt: {prompt}")
        if hasattr(e, 'response'):
            logging.error(f"Response Status Code: {e.response.status_code}")
            logging.error(f"Response Content: {e.response.text}")
        return create_mock_image()  # Return a mock image if API request fails

def create_mock_image():
    # Create a simple mock image for testing
    mock_image = Image.new('RGB', (512, 512), color='white')
    draw = ImageDraw.Draw(mock_image)
    draw.text((10, 10), "Mock Image", fill='black')
    return mock_image


def change_expression_image2image(image: Image.Image, prompt: str, api_token: str) -> Image.Image:
    if api_token == MOCK_API_TOKEN or not verify_token(api_token):
        logging.warning("Using mock image generation due to invalid API token.")
        return create_mock_image(prompt)

    try:
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_auth_token=api_token)
        pipeline = pipeline.to("cuda")
        output = pipeline(prompt=prompt, image=image, strength=0.75, guidance_scale=7.5).images[0]
        return output
    except Exception as e:
        logging.error(f"Error in change_expression_image2image: {str(e)}")
        return create_mock_image(prompt)

def main(
    output_dir: str = "output",
    api_token: Optional[str] = None,
    input_image_path: Optional[str] = None
):
    api_token = api_token or os.environ.get("HF_TOKEN") or MOCK_API_TOKEN
    use_mock = api_token == MOCK_API_TOKEN

    if not use_mock and not verify_token(api_token):
        logging.warning("Token verification failed. Falling back to mock mode.")
        use_mock = True
        api_token = MOCK_API_TOKEN

    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {api_token}"}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    def generate_and_save(prompt: str, filename: str) -> Optional[Image.Image]:
        try:
            if use_mock:
                output_image = create_mock_image(prompt)
            else:
                output_image = generate_expression(api_url, headers, prompt)
            output_path = os.path.join(output_dir, filename)
            output_image.save(output_path)
            logging.info(f"Generated {filename}: {output_path}")
            return output_image
        except requests.RequestException as e:
            logging.error(f"API Error generating {filename}: {str(e)}")
        except IOError as e:
            logging.error(f"Error saving {filename}: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error generating {filename}: {str(e)}")
        return None

    # Define expressions and their corresponding prompts
    expressions = {
        "Smiling": "a character with a warm, genuine smile",
        "Laughing": "a character laughing heartily",
        "Surprised": "a character with a surprised expression, raised eyebrows",
        "Sad": "a character with a sad, downcast expression",
        "Mad": "a character with an angry, frustrated expression",
        "Afraid": "a character with a fearful, wide-eyed expression"
    }

    if input_image_path:
        # Load the input image
        input_image = Image.open(input_image_path)

        # Generate images for each expression using image2image
        for expression, prompt in expressions.items():
            output_image = change_expression_image2image(input_image, prompt, api_token, use_mock)
            output_path = os.path.join(output_dir, f"{expression.lower()}_modified.png")
            output_image.save(output_path)
            logging.info(f"Generated modified {expression}: {output_path}")
    else:
        # Generate images for each expression from scratch
        for expression, prompt in expressions.items():
            generate_and_save(prompt, f"{expression.lower()}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multiple expressions for a 2D character illustration")
    parser.add_argument("--output_dir", default="output", help="Directory to save output images")
    parser.add_argument("--api_token", help="Hugging Face API token")
    parser.add_argument("--input_image", help="Path to input image for expression modification")
    args = parser.parse_args()

    main(args.output_dir, args.api_token, args.input_image)
