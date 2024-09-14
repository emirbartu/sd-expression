import os
import logging
from typing import Optional
from dotenv import load_dotenv
import requests
from PIL import Image
import io
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from transformers import CLIPTokenizer
from huggingface_hub import hf_hub_download
import tempfile

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hugging Face API constants
HF_API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Global variable to store the pipeline
pipeline_cache = None

def verify_token(api_token: str) -> bool:
    headers = {"Authorization": f"Bearer {api_token}"}
    try:
        response = requests.get(HF_API_URL, headers=headers)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        logging.error(f"Token verification failed: {str(e)}")
        return False

def resize_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    width, height = image.size
    if width > height:
        if width > max_size:
            height = int(height * (max_size / width))
            width = max_size
    else:
        if height > max_size:
            width = int(width * (max_size / height))
            height = max_size
    return image.resize((width, height), Image.LANCZOS)

def load_pipeline(api_token: str) -> StableDiffusionImg2ImgPipeline:
    global pipeline_cache
    if pipeline_cache is not None:
        logging.info("Using cached pipeline")
        return pipeline_cache

    try:
        logging.info("Loading pipeline...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                use_auth_token=api_token,
                cache_dir=tmp_dir,
                torch_dtype=torch.float32
            )

        logging.info("Optimizing pipeline for CPU processing")
        pipeline = pipeline.to("cpu")
        pipeline.enable_attention_slicing(slice_size="max")

        pipeline_cache = pipeline
        logging.info("Pipeline loaded and cached successfully")
        return pipeline
    except Exception as e:
        logging.error(f"Error loading pipeline: {str(e)}")
        raise

def change_expression_image2image(image: Image.Image, pipeline: StableDiffusionImg2ImgPipeline, emotion: str) -> Image.Image:
    prompt = f"a character with a {emotion} expression, highly detailed, realistic"
    negative_prompt = "blurry, low quality, distorted"
    try:
        logging.info(f"Generating image with prompt: {prompt}")
        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=0.6,
            guidance_scale=7.0,
            num_inference_steps=20,
            num_images_per_prompt=1
        ).images[0]
        logging.info("Image generation successful")
        return output
    except Exception as e:
        logging.error(f"Error in image2image processing: {str(e)}")
        raise

def main(input_image_path: str, output_path: str, emotion: str, api_token: Optional[str] = None):
    api_token = api_token or HF_TOKEN
    if not api_token:
        raise ValueError("API token is required. Provide it as an argument or set the HF_TOKEN environment variable.")

    if not verify_token(api_token):
        logging.error("Token verification failed. Exiting.")
        return

    try:
        # Load and resize the input image
        input_image = Image.open(input_image_path)
        resized_image = resize_image(input_image)
        logging.info(f"Resized input image to {resized_image.size}")

        # Load the pipeline
        pipeline = load_pipeline(api_token)

        # Generate the image with the specified expression using image2image
        output_image = change_expression_image2image(resized_image, pipeline, emotion)
        output_image.save(output_path)
        logging.info(f"Generated {emotion} expression image: {output_path}")
    except MemoryError:
        logging.warning("Memory error. Trying with a smaller image size.")
        try:
            smaller_image = resize_image(input_image, max_size=256)
            output_image = change_expression_image2image(smaller_image, pipeline, emotion)
            output_image.save(output_path)
            logging.info(f"Generated {emotion} expression image with reduced size: {output_path}")
        except Exception as e:
            logging.error(f"Failed to generate image even with reduced size: {str(e)}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Change expression of an input image using Stable Diffusion XL")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_path", help="Path to save the output image")
    parser.add_argument("emotion", help="Desired emotion for the output image")
    parser.add_argument("--api_token", help="Hugging Face API token")
    args = parser.parse_args()

    main(args.input_image, args.output_path, args.emotion, args.api_token)
