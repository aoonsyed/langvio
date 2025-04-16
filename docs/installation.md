# Installation Guide

This guide provides detailed instructions for installing Langvio and its dependencies.

## Requirements

Langvio requires:

- Python 3.8 or later
- PyTorch 1.7.0 or later
- An API key for at least one LLM provider (OpenAI, Google, etc.)

## Basic Installation

You can install Langvio using pip:

```bash
pip install langvio
```

This installs the core package, but you'll need to install at least one LLM provider to use it.

## Installation with LLM Providers

Langvio supports multiple LLM providers through optional dependencies:

### OpenAI

```bash
pip install langvio[openai]
```

This installs:
- `openai>=1.0.0`
- `langchain-openai>=0.0.1`

### Google Gemini

```bash
pip install langvio[google]
```

This installs:
- `google-generativeai>=0.3.0`
- `langchain-google-genai>=0.0.1`

### All LLM Providers

To install all supported LLM providers:

```bash
pip install langvio[all-llm]
```

### Development Tools

For development:

```bash
pip install langvio[dev]
```

This installs:
- `pytest>=6.0.0`
- `black>=21.5b2`
- `isort>=5.9.1`
- `flake8>=3.9.2`

## Installing from Source

To install from source:

```bash
git clone https://github.com/yourusername/langvio.git
cd langvio
pip install -e ".[all-llm,dev]"
```

## YOLO Models

Langvio uses YOLOv11 models for object detection. The first time you run Langvio, it will automatically download the default model (yolov11n.pt).

If you want to use a different YOLO model, you can specify it in your configuration:

```yaml
vision:
  default: "yolo"
  models:
    yolo:
      type: "yolo"
      model_path: "yolov11x.pt"  # Using the larger model
      confidence: 0.3
```

Available YOLO models:
- `yolov11n.pt` - Small, fast model (default)
- `yolov11m.pt` - Medium model
- `yolov11x.pt` - Large, more accurate model

## Environment Setup

### API Keys

Langvio requires API keys for the LLM providers you intend to use. You can set these as environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY=your_openai_key

# For Google Gemini
export GOOGLE_API_KEY=your_google_key
```

### Using .env Files

For convenience, Langvio can load API keys from a `.env` file:

1. Create a `.env` file in your project directory
2. Add your API keys to the file:
   ```
   OPENAI_API_KEY=your_openai_key
   GOOGLE_API_KEY=your_google_key
   ```
3. Langvio will automatically load these when imported

## Troubleshooting Installation

### Missing LLM Provider

If you see an error like:
```
ERROR: No LLM providers are installed. Please install at least one provider.
```

You need to install at least one LLM provider:
```bash
pip install langvio[openai]  # or another provider
```

### CUDA Issues

If you're having issues with CUDA and PyTorch:

1. Ensure you have a compatible CUDA version installed
2. Install PyTorch with the appropriate CUDA version:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
   ```

### Permission Issues

If you encounter permission errors during installation:

#### On Unix/Linux/MacOS:
```bash
pip install --user langvio[all-llm]
```

#### On Windows:
Run the command prompt as administrator.

## Verifying Installation

You can verify your installation by running a simple example:

```python
import langvio

# List available processors
from langvio import registry
print("Available LLM processors:", registry.list_llm_processors())
print("Available vision processors:", registry.list_vision_processors())
```

If your installation is correct, this should output the available processors without errors.