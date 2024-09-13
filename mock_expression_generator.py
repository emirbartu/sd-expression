import argparse
from PIL import Image, ImageDraw, ImageEnhance

def load_image(image_path):
    try:
        with Image.open(image_path) as img:
            return img.copy()
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def apply_expression(image, expression):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    if expression == "smiling":
        # Draw a curved line for a smile
        draw.arc([width*0.3, height*0.6, width*0.7, height*0.8], start=0, end=180, fill="black", width=3)
    elif expression == "laughing":
        # Draw a wider smile and squint eyes
        draw.arc([width*0.25, height*0.6, width*0.75, height*0.9], start=0, end=180, fill="black", width=4)
        draw.line([width*0.3, height*0.4, width*0.4, height*0.35], fill="black", width=2)
        draw.line([width*0.6, height*0.35, width*0.7, height*0.4], fill="black", width=2)
    elif expression == "surprised":
        # Draw an open mouth and raised eyebrows
        draw.ellipse([width*0.4, height*0.6, width*0.6, height*0.75], outline="black", width=2)
        draw.arc([width*0.3, height*0.3, width*0.45, height*0.4], start=0, end=180, fill="black", width=2)
        draw.arc([width*0.55, height*0.3, width*0.7, height*0.4], start=0, end=180, fill="black", width=2)
    elif expression == "sad":
        # Draw a frown and droopy eyes
        draw.arc([width*0.3, height*0.7, width*0.7, height*0.9], start=180, end=0, fill="black", width=3)
        draw.line([width*0.3, height*0.45, width*0.4, height*0.5], fill="black", width=2)
        draw.line([width*0.6, height*0.5, width*0.7, height*0.45], fill="black", width=2)
    elif expression == "mad":
        # Draw angry eyebrows and a stern mouth
        draw.line([width*0.3, height*0.35, width*0.45, height*0.4], fill="black", width=3)
        draw.line([width*0.55, height*0.4, width*0.7, height*0.35], fill="black", width=3)
        draw.line([width*0.35, height*0.7, width*0.65, height*0.7], fill="black", width=3)
    elif expression == "afraid":
        # Draw wide eyes and an open mouth
        draw.ellipse([width*0.35, height*0.35, width*0.45, height*0.45], outline="black", width=2)
        draw.ellipse([width*0.55, height*0.35, width*0.65, height*0.45], outline="black", width=2)
        draw.ellipse([width*0.4, height*0.6, width*0.6, height*0.75], outline="black", width=2)

    return image

def generate_expressions(input_image):
    expressions = ["smiling", "laughing", "surprised", "sad", "mad", "afraid"]
    results = []

    for expression in expressions:
        modified_image = input_image.copy()
        result = apply_expression(modified_image, expression)
        results.append((expression, result))

    return results

def main(input_image_path):
    input_image = load_image(input_image_path)
    if input_image is None:
        return

    results = generate_expressions(input_image)

    for expression, image in results:
        output_filename = f"{expression}_output.png"
        image.save(output_filename)
        print(f"Generated {expression} expression: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mock expressions for a 2D character")
    parser.add_argument("input_image", help="Path to the input image")
    args = parser.parse_args()

    main(args.input_image)
