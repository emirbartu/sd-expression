import argparse
import requests
import io
import base64
from PIL import Image
import os
import sys
from typing import Union, Optional

def verify_token(api_token: str) -> bool:
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {api_token}"}
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Token verification failed: {str(e)}")
        return False

def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")

def encode_image(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    width, height = image.size
    if width > max_size or height > max_size:
        ratio = max(width, height) / max_size
        new_size = (int(width / ratio), int(height / ratio))
        return image.resize(new_size, Image.LANCZOS)
    return image

def generate_expression(
    api_url: str,
    headers: dict,
    init_image: Image.Image,
    prompt: str
) -> Image.Image:
    # Encode the image to base64
    buffered = io.BytesIO()
    init_image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Prepare the payload
    payload = {
        "inputs": [
            {
                "image": encoded_image,
                "prompt": prompt
            }
        ]
    }

    # Update headers
    headers.update({"Content-Type": "application/json"})

    try:
        # Print the full request payload for debugging (excluding the image data)
        debug_payload = {**payload, "inputs": [{"prompt": prompt, "image": "<base64_image_data>"}]}
        print(f"Request Payload: {debug_payload}")

        # Send the request with JSON payload
        response = requests.post(
            api_url,
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        # Process the response
        if response.headers.get('content-type') == 'application/json':
            response_data = response.json()
            if isinstance(response_data, list) and len(response_data) > 0:
                # Assuming the API returns a list with the first item being the image
                image_data = base64.b64decode(response_data[0])
                output_image = Image.open(io.BytesIO(image_data))
                return output_image
            else:
                error_msg = response_data.get('error', 'Unknown error occurred')
                raise requests.RequestException(f"API Error: {error_msg}")
        else:
            raise requests.RequestException("Unexpected response format")
    except requests.RequestException as e:
        print(f"Error in generate_expression: {str(e)}")
        print(f"Request URL: {api_url}")
        print(f"Request Headers: {headers}")
        print(f"Request Prompt: {prompt}")
        if hasattr(e, 'response'):
            print(f"Response Status Code: {e.response.status_code}")
            print(f"Response Content: {e.response.text}")
        raise

def load_html_input(html_file: str) -> str:
    with open(html_file, 'r') as f:
        content = f.read()
    # Extract the image path from the HTML content
    # This is a simple implementation and might need to be adjusted based on the HTML structure
    import re
    match = re.search(r'<img\s+src=["\'](.+?)["\']', content)
    if match:
        return match.group(1)
    raise ValueError("No image found in HTML file")

def generate_angry_pig(api_url: str, headers: dict) -> Image.Image:
    prompt = "An angry cartoon pig, 2D illustration, vibrant colors, expressive face"
    headers.update({"Content-Type": "application/json"})
    payload = {
        "inputs": [
            {
                "prompt": prompt
            }
        ]
    }

    print(f"Request payload for generate_angry_pig: {payload}")

    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        if response.headers.get('content-type') == 'application/json':
            response_data = response.json()
            if isinstance(response_data, list) and len(response_data) > 0:
                image_data = base64.b64decode(response_data[0])
                return Image.open(io.BytesIO(image_data))
            else:
                error_msg = response_data.get('error', 'Unknown error occurred')
                print(f"API Error: {error_msg}")
                print(f"Full API Response: {response_data}")
                raise requests.RequestException(f"API Error: {error_msg}")
        else:
            raise requests.RequestException("Unexpected response format")
    except requests.RequestException as e:
        print(f"Error in generate_angry_pig: {str(e)}")
        print(f"Request URL: {api_url}")
        print(f"Request Headers: {headers}")
        print(f"Request Prompt: {prompt}")
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Content: {response.text}")
        raise

def transform_to_happy_pig(api_url: str, headers: dict, angry_pig: Image.Image) -> Image.Image:
    prompt = "Transform the angry pig into a happy, smiling cartoon pig, maintain consistent features, 2D illustration"
    return generate_expression(api_url, headers, angry_pig, prompt)

def main(
    input_image: Union[str, Image.Image],
    output_dir: str = "output",
    api_token: Optional[str] = None,
    html_input: Optional[str] = None
):
    if html_input:
        input_image = load_html_input(html_input)

    default_token = "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"
    api_token = api_token or os.environ.get("HF_API_TOKEN") or default_token
    if not api_token:
        raise ValueError("API token is required. Provide it as an argument, set the HF_API_TOKEN environment variable, or use the default token.")

    # Verify the token before proceeding
    if not verify_token(api_token):
        print("Token verification failed. Exiting.")
        sys.exit(1)

    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {api_token}"}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    def generate_and_save(prompt: str, filename: str, input_image: Optional[Image.Image] = None) -> Optional[Image.Image]:
        try:
            if input_image:
                output_image = generate_expression(api_url, headers, input_image, prompt)
            else:
                output_image = generate_angry_pig(api_url, headers)
            output_path = os.path.join(output_dir, filename)
            output_image.save(output_path)
            print(f"Generated {filename}: {output_path}")
            return output_image
        except requests.exceptions.RequestException as e:
            print(f"Error generating {filename}: {str(e)}")
            return None

    # Generate angry pig cartoon
    angry_pig = generate_and_save("An angry cartoon pig, 2D illustration, vibrant colors, expressive face", "angry_pig.png")

    # Transform angry pig to happy pig
    if angry_pig:
        generate_and_save("Transform the angry pig into a happy, smiling cartoon pig, maintain consistent features, 2D illustration", "happy_pig.png", angry_pig)

    # Load and resize the input image
    if isinstance(input_image, str):
        init_image = load_image(input_image)
    else:
        init_image = input_image
    init_image = resize_image(init_image)

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
        generate_and_save(prompt, f"{expression.lower()}.png", init_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multiple expressions for a 2D character illustration")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("--output_dir", default="output", help="Directory to save output images")
    parser.add_argument("--api_token", help="Hugging Face API token")
    parser.add_argument("--html_input", help="Path to HTML file containing image input")
    args = parser.parse_args()

    main(args.input_image, args.output_dir, args.api_token, args.html_input)
