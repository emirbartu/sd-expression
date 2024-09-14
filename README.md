# AI Consistent Character Expression Generation

This project generates consistent character expressions for 2D illustrations using AI-powered image generation techniques. It leverages the Hugging Face Inference API with the Stable Diffusion XL model to create high-quality, diverse expressions while maintaining character consistency.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Generated Expressions](#generated-expressions)
5. [Configuration](#configuration)
6. [Background Removal](#background-removal)
7. [Future Development](#future-development)
8. [Contributing](#contributing)
9. [License](#license)

## Project Overview

The AI Consistent Character Expression Generation pipeline takes a 2D character illustration as input and generates six different expressions: smiling, laughing, surprised, sad, mad, and afraid. The project uses the Stable Diffusion XL model via the Hugging Face Inference API to create high-quality, AI-powered expressions.

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
     HUGGINGFACE_API_TOKEN=your_api_token_here
     ```

## Usage

To use the character expression generation script:

1. Ensure you have set up your Hugging Face API token in the `.env` file.
2. Prepare a 2D character illustration in PNG format.
3. Run the script:
   ```
   python main.py path/to/your/input_image.png
   ```
4. The script will generate six images with different expressions in the "generated_expressions" folder.

Optional: You can specify the maximum number of retries for API calls:
```
python main.py path/to/your/input_image.png --max_retries 5
```

## Generated Expressions

The project includes a "generated_expressions" folder in the main branch of the repository. This folder is used to store the images generated by the script. Each time you run the script, it will create six new images (one for each expression) and save them in this folder. This allows you to easily access and manage the generated expressions for your characters.

## Configuration

The script uses a configuration dictionary (`CONFIG`) to set various parameters for the image generation process. You can modify these settings in the `main.py` file:

- `HF_MODEL`: The Hugging Face model to use (default: "stabilityai/stable-diffusion-xl-base-1.0")
- `STEPS`: Number of inference steps (default: 30)
- `GUIDANCE_SCALE`: Guidance scale for generation (default: 7.5)
- `STRENGTH`: Strength of the transformation (default: 0.7)
- `SAMPLER`: Sampling method (default: "lcm")
- `SCHEDULER`: Scheduler type (default: "normal")

## Background Removal

The script includes a basic background removal function using K-means clustering. This helps isolate the character from the background in the generated images. The background removal process is automatically applied to each generated expression.

## Future Development

- Improve the background removal algorithm for better accuracy.
- Implement fine-tuning options for better consistency across generated expressions.
- Add support for custom prompts and more diverse expression types.
- Enhance error handling and logging for better debugging.

## Contributing

Contributions to this project are welcome! Please feel free to submit issues, feature requests, or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
