import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# Configuration
CONFIG = {
    "HF_MODEL": "stabilityai/stable-diffusion-xl-base-1.0",
    "STEPS": 30,
    "GUIDANCE_SCALE": 7.5,
    "SEED": 152,
}

class ImageGenerator:
    def __init__(self, model_id=CONFIG["HF_MODEL"], device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    def generate_image(self, prompt, negative_prompt="", seed=CONFIG["SEED"]):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=CONFIG["STEPS"],
            guidance_scale=CONFIG["GUIDANCE_SCALE"],
            generator=generator
        ).images[0]
        return image

    def save_image(self, image, filename, output_dir="generated_images"):
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        image.save(output_path)
        print(f"Image saved: {output_path}")

def main():
    generator = ImageGenerator()
    prompt = "A beautiful landscape with mountains and a lake"
    image = generator.generate_image(prompt)
    generator.save_image(image, "generated_landscape.png")

if __name__ == "__main__":
    main()
