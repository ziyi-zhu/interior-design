# Interior Design AI Assistant

An AI-powered interior design tool combining Stable Diffusion 3.5 and Llama 3.2 11B Vision to transform living spaces.

## Overview

This project uses state-of-the-art AI models to analyze interior spaces and generate redesign concepts. Llama 3.2 11B Vision extracts features from existing interiors, while Stable Diffusion 3.5 generates photorealistic redesigns using ControlNet.

## Features

- **Image Analysis**: Llama 3.2 11B Vision analyzes your interior photos
- **Smart Prompt Generation**: Automatically creates tailored prompts for redesign
- **Advanced Image Processing**: Supports Canny edge detection, blur, and depth map conditioning
- **High-Quality Visualization**: Generates photorealistic interior designs using SD 3.5
- **Customizable Configuration**: Adjust image dimensions, guidance scale, and other parameters

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/interior-design.git
cd interior-design

# Set up a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up pre-commit hooks
pre-commit install
```

## Usage

1. Place your interior photos in the `inputs` directory
2. Run the application:

```bash
python main.py
```

By default, this uses Canny edge detection and saves results to the `outputs/canny` directory.

## Configuration

Modify parameters in `main.py` to adjust the design process:

```python
config = DesignConfig(
    input_dir='inputs',
    output_dir='outputs/canny',
    transform_type='canny',  # Options: 'canny', 'blur', 'depth'
    # Additional parameters available in DesignConfig class
)
```

## Development

### Code Formatting

This project uses [pre-commit](https://pre-commit.com/) to maintain code quality. The following tools are configured:

- **black**: Code formatter
- **isort**: Import sorter
- **flake8**: Code linter
- **mypy**: Type checker

To format your code:

```bash
# Format all files in the repository
pre-commit run --all-files

# Format automatically on commit
# (This happens automatically after running `pre-commit install`)
```

If you want to run formatters individually:

```bash
# Format with black
black .

# Sort imports with isort
isort .

# Check with flake8
flake8 .

# Type check with mypy
mypy .
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU recommended

## Technologies

- **Stable Diffusion 3.5**: For generating interior designs
- **Llama 3.2 11B Vision**: For analyzing interior spaces
- **ControlNet**: For conditional image generation
