import os
import logging
from typing import Optional
from dotenv import load_dotenv
import requests
from PIL import Image
import io

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

def generate_image(prompt: str, api_token: str) -> Optional[Image.Image]:
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt}

    logging.info(f"Sending request to {HF_API_URL}")
    logging.info(f"Headers: {headers}")
    logging.info(f"Payload: {payload}")

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response headers: {response.headers}")

        response.raise_for_status()

        content_type = response.headers.get('content-type')
        logging.info(f"Response content type: {content_type}")

        if content_type in ['image/png', 'image/jpeg']:
            return Image.open(io.BytesIO(response.content))
        else:
            logging.error(f"Unexpected response format. Content: {response.text}")
            raise requests.RequestException(f"Unexpected response format: {content_type}")
    except requests.RequestException as e:
        logging.error(f"Error in generate_image: {str(e)}")
        if hasattr(e, 'response'):
            logging.error(f"Response content: {e.response.text}")
        return None

def main(prompt: str, output_path: str, api_token: Optional[str] = None):
    api_token = api_token or HF_TOKEN
    if not api_token:
        raise ValueError("API token is required. Provide it as an argument or set the HF_TOKEN environment variable.")

    if not verify_token(api_token):
        logging.error("Token verification failed. Exiting.")
        return

    image = generate_image(prompt, api_token)
    if image:
        image.save(output_path)
        logging.info(f"Image generated and saved to: {output_path}")
    else:
        logging.error("Failed to generate image.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate an image using Stable Diffusion")
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("output_path", help="Path to save the generated image")
    parser.add_argument("--api_token", help="Hugging Face API token")
    args = parser.parse_args()

    main(args.prompt, args.output_path, args.api_token)
