# AI Consistent Character Generation Pipeline

This project aims to generate consistent character expressions for 2D illustrations using AI-powered image generation techniques. It includes two main scripts: a mock expression generator for proof of concept and a more advanced script using the Hugging Face Inference API.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Mock Expression Generator](#mock-expression-generator)
   - [Hugging Face API Script](#hugging-face-api-script)
4. [Future Development](#future-development)
5. [Contributing](#contributing)
6. [License](#license)

## Project Overview

The AI Consistent Character Generation Pipeline is designed to take a 2D character illustration as input and generate six different expressions: smiling, laughing, surprised, sad, mad, and afraid. The project consists of two main components:

1. **Mock Expression Generator**: A proof-of-concept script that applies simple image manipulations to create different expressions.
2. **Hugging Face API Script**: An advanced script that uses the Stable Diffusion XL model via the Hugging Face Inference API to generate high-quality, AI-powered expressions.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/emirbartu/sd-expression.git
   cd sd-expression
   ```

2. Install the required dependencies:
   ```
   pip install pillow requests
   ```

3. (Optional) For the Hugging Face API script, set up your API token as an environment variable:
   ```
   export HUGGINGFACE_API_TOKEN=your_api_token_here
   ```

## Usage

### Mock Expression Generator

The mock expression generator is a simple script that applies basic image manipulations to create different expressions. It's useful for testing the pipeline and as a fallback option.

To use the mock expression generator:

1. Prepare a 2D character illustration in PNG format.
2. Run the script:
   ```
   python mock_expression_generator.py path/to/your/input_image.png
   ```
3. The script will generate six images with different expressions in the current directory.

### Hugging Face API Script

The Hugging Face API script uses the Stable Diffusion XL model to generate high-quality expressions. Note that this script requires a valid Hugging Face API token to function.

To use the Hugging Face API script:

1. Ensure you have set up your Hugging Face API token as an environment variable (see Installation step 3).
2. Prepare a 2D character illustration in PNG format.
3. Run the script:
   ```
   python comfyui_api_script.py path/to/your/input_image.png
   ```
4. The script will generate six images with different expressions in the current directory.

Note: If you encounter any issues with the Hugging Face API script, make sure your API token is valid and you have sufficient quota for API calls.

## Future Development

- Implement background removal functionality in the Hugging Face API script.
- Integrate with ComfyUI for local processing and more advanced workflows.
- Improve consistency and quality of generated expressions.

## Contributing

Contributions to this project are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
