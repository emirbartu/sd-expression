import argparse
import base64
from io import BytesIO
from PIL import Image
import requests
import logging
import os
import sys
import time
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from sklearn.cluster import KMeans

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
    "IMAGE_CFG_SCALE": 1.5,
    "PROMPT_STRENGTH": 0.8,
}

# Hugging Face API constants
HF_API_URL = f"https://api-inference.huggingface.co/models/{CONFIG['HF_MODEL']}"
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
        # Open the image again and return it as a PIL Image object
        return Image.open(image_path)
    except (IOError, OSError, Image.DecompressionBombError) as e:
        logging.error(f"Error loading image {image_path}: {str(e)}")
        return None

# The get_available_models function is removed as it's not needed for the Hugging Face API.
# We're using a specific model (Stable Diffusion XL) directly in this script.

def generate_expression(expression, input_image, max_retries=3, timeout=60):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    # Prepare the payload for image2image
    payload = {
        "inputs": f"portrait of a 2D character, {expression} expression, consistent features, detailed, high quality",
        "negative_prompt": "low quality, blurry, distorted features, inconsistent style",
        "num_inference_steps": CONFIG["STEPS"],
        "guidance_scale": CONFIG["GUIDANCE_SCALE"],
        "strength": CONFIG["STRENGTH"],
        "scheduler": CONFIG["SCHEDULER"],
        "seed": CONFIG["SEED"],
    }

    # Add the input image to the payload for img2img
    buffered = BytesIO()
    input_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    payload["image"] = img_str

    for attempt in range(max_retries):
        try:
            logging.info(f"Generating {expression} expression (attempt {attempt + 1}/{max_retries})...")
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()

            # Check if the response is JSON (error) or image data
            if response.headers.get('content-type') == 'application/json':
                error_data = response.json()
                raise requests.RequestException(f"API Error: {error_data.get('error', 'Unknown error')}")

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
    # Color-based segmentation for background removal
    from PIL import Image
    import numpy as np
    from sklearn.cluster import KMeans

    # Convert image to numpy array
    img_array = np.array(image)

    # Reshape the image to be a list of pixels
    pixels = img_array.reshape((-1, 3))

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(pixels)

    # Create a mask based on the cluster labels
    mask = kmeans.labels_.reshape(img_array.shape[:2])

    # Determine which cluster represents the background
    # Assume the cluster with more pixels is the background
    background_label = 0 if np.sum(mask == 0) > np.sum(mask == 1) else 1

    # Create an alpha channel
    alpha = np.where(mask == background_label, 0, 255).astype(np.uint8)

    # Add alpha channel to the image
    result = np.dstack((img_array, alpha))

    # Convert back to PIL Image
    return Image.fromarray(result, 'RGBA')

def main(input_image_path, max_retries=3, remove_bg=False):
    expressions = ["smiling", "laughing", "surprised", "sad", "mad", "afraid"]
    generated_images = []

    # Create the output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load the input image
    input_image = load_image(input_image_path)
    if input_image is None:
        logging.error(f"Failed to load input image: {input_image_path}")
        return []

    # LoRA and VAE models are handled by the Hugging Face API
    # No need to load them locally

    for expression in expressions:
        success = False
        for attempt in range(max_retries):
            try:
                logging.info(f"Generating {expression} expression (attempt {attempt + 1}/{max_retries})...")
                generated_image = generate_expression(expression, input_image)
                if generated_image:
                    if remove_bg:
                        processed_image = remove_background(generated_image)
                        output_filename = os.path.join(OUTPUT_FOLDER, f"{expression}_output_nobg.png")
                    else:
                        processed_image = generated_image
                        output_filename = os.path.join(OUTPUT_FOLDER, f"{expression}_output.png")

                    processed_image.save(output_filename)
                    generated_images.append(output_filename)
                    logging.info(f"Generated and processed {expression} expression: {output_filename}")
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
        logging.info(f"All {total_expressions} expressions generated and processed successfully.")
    else:
        logging.warning(f"Generated and processed {generated_count} out of {total_expressions} expressions.")

    return generated_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate consistent character expressions")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for API calls")
    parser.add_argument("--output_dir", help="Directory to save output images")
    parser.add_argument("--remove_background", action="store_true", help="Enable background removal")
    args = parser.parse_args()

    # Update OUTPUT_FOLDER with the user-specified directory if provided
    if args.output_dir:
        global OUTPUT_FOLDER
        OUTPUT_FOLDER = args.output_dir

    main(args.input_image, max_retries=args.max_retries, remove_bg=args.remove_background)
