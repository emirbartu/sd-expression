import argparse
import base64
import io
from io import BytesIO
from PIL import Image
import requests
import json
import logging
import os
import sys
import time
from typing import Optional
from dotenv import load_dotenv

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

def verify_token(api_token: str) -> bool:
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
        raise



def main(
    output_dir: str = "output",
    api_token: Optional[str] = None
):
    api_token = api_token or os.environ.get("HF_TOKEN")
    if not api_token:
        raise ValueError("API token is required. Provide it as an argument, set the HF_TOKEN environment variable, or use the default token.")

    # Verify the token before proceeding
    if not verify_token(api_token):
        logging.error("Token verification failed. Exiting.")
        sys.exit(1)

    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {api_token}"}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    def generate_and_save(prompt: str, filename: str) -> Optional[Image.Image]:
        try:
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

    # Generate images for each expression
    for expression, prompt in expressions.items():
        generate_and_save(prompt, f"{expression.lower()}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multiple expressions for a 2D character illustration")
    parser.add_argument("--output_dir", default="output", help="Directory to save output images")
    parser.add_argument("--api_token", help="Hugging Face API token")
    args = parser.parse_args()

    main(args.output_dir, args.api_token)
