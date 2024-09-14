# AI Consistent Character Expression Generation

This project generates consistent character expressions for 2D illustrations using AI-powered image generation techniques. It leverages the Stable Diffusion XL model to create high-quality, diverse expressions while maintaining character consistency.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Image Generation](#image-generation)
   - [Expression Change](#expression-change)
4. [Configuration](#configuration)
5. [Background Removal](#background-removal)
6. [Future Development](#future-development)
7. [Contributing](#contributing)
8. [License](#license)

## Project Overview

The AI Consistent Character Expression Generation project consists of two main components:
1. Image Generation: Creates new images based on text prompts.
2. Expression Change: Modifies existing character images to display different expressions.

The project uses the Stable Diffusion XL model to create high-quality, AI-powered expressions.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/emirbartu/sd-expression.git
   cd sd-expression
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Hugging Face API token:
   - Create a `.env` file in the project root directory
   - Add your Hugging Face API token to the `.env` file:
     ```
     HUGGINGFACE_API_TOKEN=YOUR_API_TOKEN
     ```

## Usage

### Image Generation

To generate a new image based on a text prompt:

1. Run the `image_generation.py` script:
   ```
   python image_generation.py
   ```

2. The script will generate an image based on the default prompt and save it in the "generated_images" folder.

To customize the image generation:

1. Open `image_generation.py` in a text editor.
2. Modify the `prompt` variable in the `main()` function with your desired text.
3. Run the script as described above.

### Expression Change

To change the expression of an existing character image:

1. Prepare a 2D character illustration in PNG format.
2. Run the `expression_change.py` script:
   ```
   python expression_change.py path/to/your/input_image.png
   ```
3. The script will generate six images with different expressions (smiling, laughing, surprised, sad, mad, afraid) in the "output" folder.

Optional: You can specify a custom output directory and enable background removal:
```
python expression_change.py path/to/your/input_image.png --output_dir custom_output --remove_background
```

## Configuration

Both scripts use a configuration dictionary (`CONFIG`) to set various parameters for the image generation process. You can modify these settings in the respective Python files:

- `HF_MODEL`: The Hugging Face model to use (default: "stabilityai/stable-diffusion-xl-base-1.0")
- `STEPS`: Number of inference steps (default: 30)
- `GUIDANCE_SCALE`: Guidance scale for generation (default: 7.5)
- `STRENGTH`: Strength of the transformation (default: 0.7, only for expression change)
- `SEED`: Random seed for reproducibility (default: 152)

## Background Removal

The `expression_change.py` script includes an optional background removal feature. This helps isolate the character from the background in the generated images. To enable background removal, use the `--remove_background` flag when running the script.

## Future Development

- Implement a user-friendly interface for easier interaction with the scripts.
- Add support for batch processing of multiple input images.
- Enhance the background removal algorithm for better accuracy.
- Implement fine-tuning options for better consistency across generated expressions.
- Add support for custom prompts and more diverse expression types.
- Enhance error handling and logging for better debugging.

## Contributing

Contributions to this project are welcome! Please feel free to submit issues, feature requests, or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
