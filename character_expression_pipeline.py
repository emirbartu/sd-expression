import argparse
import requests
import base64
import io
from PIL import Image
import os
import sys
from typing import Optional
from remove_background import remove_background

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
        print(f"Request Payload: {payload}")

        # Send the request with JSON payload
        response = requests.post(
            api_url,
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        # Process the response
        if response.headers.get('content-type') == 'image/png':
            output_image = Image.open(io.BytesIO(response.content))
            return output_image
        elif response.headers.get('content-type') == 'application/json':
            response_data = response.json()
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



# The generate_angry_pig function has been removed as it's no longer needed.

# Function removed as it's no longer needed without input images.

def main(
    output_dir: str = "output",
    api_token: Optional[str] = None,
    remove_background: bool = False
):
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

    def generate_and_save(prompt: str, filename: str, remove_bg: bool = False) -> Optional[Image.Image]:
        try:
            output_image = generate_expression(api_url, headers, prompt)
            if remove_bg:
                output_image = remove_background(output_image)
            output_path = os.path.join(output_dir, filename)
            output_image.save(output_path)
            print(f"Generated {filename}: {output_path}")
            return output_image
        except requests.RequestException as e:
            print(f"API Error generating {filename}: {str(e)}")
        except IOError as e:
            print(f"Error saving {filename}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error generating {filename}: {str(e)}")
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
        generate_and_save(prompt, f"{expression.lower()}.png", remove_background)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multiple expressions for a 2D character illustration")
    parser.add_argument("--output_dir", default="output", help="Directory to save output images")
    parser.add_argument("--api_token", help="Hugging Face API token")
    parser.add_argument("--remove_background", action="store_true", help="Enable background removal")
    args = parser.parse_args()

    main(args.output_dir, args.api_token, args.remove_background)
