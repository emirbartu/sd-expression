import argparse
import base64
from io import BytesIO
from PIL import Image
import requests
import json
import logging
import os
import sys
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hugging Face API constants
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")

if not HF_API_TOKEN:
    logging.error("HUGGINGFACE_API_TOKEN environment variable is not set. Please set it with your API token.")
    sys.exit(1)

# Headers for Hugging Face API requests
HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

def load_image(image_path):
    try:
        with Image.open(image_path) as img:
            # Verify that the image can be opened
            img.verify()
        return image_path
    except (IOError, OSError, Image.DecompressionBombError) as e:
        logging.error(f"Error loading image {image_path}: {str(e)}")
        return None

# The get_available_models function is removed as it's not needed for the Hugging Face API.
# We're using a specific model (Stable Diffusion XL) directly in this script.

def generate_expression(expression, max_retries=3, timeout=60):
    if not HF_API_TOKEN:
        logging.error("Hugging Face API token is not set. Please set the HUGGINGFACE_API_TOKEN environment variable.")
        return None

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    payload = {
        "inputs": f"portrait of a 2D character, {expression} expression, consistent features, detailed, high quality",
    }

    for attempt in range(max_retries):
        try:
            logging.info(f"Generating {expression} expression (attempt {attempt + 1}/{max_retries})...")
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content))
            logging.info(f"Successfully generated image for expression: {expression}")
            return image

        except requests.exceptions.RequestException as e:
            logging.error(f"API request error during attempt {attempt + 1}: {str(e)}")
        except Image.UnidentifiedImageError:
            logging.error(f"Received invalid image data from API for expression: {expression}")
        except Exception as e:
            logging.error(f"Unexpected error in generate_expression: {str(e)}", exc_info=True)

        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            logging.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            logging.error(f"Max retries reached. Failed to generate image for expression: {expression}")

    return None

def remove_background(image):
    # Placeholder for background removal
    # In a real implementation, you would use a more sophisticated method
    # For now, we'll just return the original image
    return image

def main(max_retries=3):
    expressions = ["smiling", "laughing", "surprised", "sad", "mad", "afraid"]
    generated_images = []

    for expression in expressions:
        success = False
        for attempt in range(max_retries):
            try:
                logging.info(f"Generating {expression} expression (attempt {attempt + 1}/{max_retries})...")
                generated_image = generate_expression(expression)
                if generated_image:
                    output_filename = f"{expression}_output.png"
                    generated_image.save(output_filename)
                    generated_images.append(output_filename)
                    logging.info(f"Generated {expression} expression: {output_filename}")
                    success = True
                    break
                else:
                    logging.warning(f"Failed to generate {expression} expression (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logging.error(f"Error generating {expression} expression (attempt {attempt + 1}/{max_retries}): {str(e)}", exc_info=True)

        if not success:
            logging.error(f"Failed to generate {expression} expression after {max_retries} attempts. Moving to next expression.")

    total_expressions = len(expressions)
    generated_count = len(generated_images)
    if generated_count == total_expressions:
        logging.info(f"All {total_expressions} expressions generated successfully.")
    else:
        logging.warning(f"Generated {generated_count} out of {total_expressions} expressions.")

    return generated_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate consistent character expressions")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for API calls")
    args = parser.parse_args()

    main(max_retries=args.max_retries)
