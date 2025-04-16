# Langvio: Connecting Language Models to Vision Models

Langvio is a Python framework that connects language models (LLMs) with vision models to enable natural language visual analysis. This library makes it easy to analyze images (and videos) using natural language queries.

## Features

- **Natural Language Interface**: Query visual content using natural language
- **Multimodal Integration**: Seamlessly connects LLMs with vision models
- **Flexible Architecture**: Supports multiple LLM providers (OpenAI, Google Gemini)
- **Rich Analysis**: Detect objects, attributes, spatial relationships, and more
- **YOLO Integration**: Powered by YOLOv11 for fast and accurate object detection
- **Extensible Design**: Easy to add new models and capabilities

## Installation

### Basic Installation

```bash
pip install langvio
```

### With LLM Providers

Langvio supports multiple LLM providers. Install the ones you need:

```bash
# For OpenAI models
pip install langvio[openai]

# For Google Gemini models
pip install langvio[google]

# For all supported providers
pip install langvio[all-llm]

# For development
pip install langvio[dev]
```

## Quick Start

```python
import os
from langvio import create_pipeline

# Create a pipeline
pipeline = create_pipeline()

# Create output directory
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Process an image with a natural language query
image_path = "path/to/your/image.jpg"
query = "What objects are in this image?"

# Run the query
result = pipeline.process(query, image_path)

# View the results
print(f"Explanation: {result['explanation']}")
print(f"Output saved to: {result['output_path']}")
```

## Using Environment Variables

Langvio supports loading API keys from a `.env` file:

1. Create a `.env` file in your project directory:

```bash
# Copy the template
cp .env.template .env

# Edit the file with your actual API keys
nano .env  # or use your preferred editor
```

2. Add your API keys:

```
GOOGLE_API_KEY=your_actual_google_api_key_here
OPENAI_API_KEY=your_actual_openai_api_key_here
# Add other API keys as needed
```

3. Langvio will automatically load these environment variables:

```python
import langvio

# API keys are automatically loaded from .env file
pipeline = langvio.create_pipeline()
```

**Important**: Add `.env` to your `.gitignore` file to prevent accidentally committing your API keys:

```bash
echo ".env" >> .gitignore
```

## Query Types

Langvio supports various types of visual analysis:

### Basic Object Detection

```python
query = "What objects are in this image?"
```

### Object Counting

```python
query = "Count how many people are in this image"
```

### Attribute Detection

```python
query = "Find all red objects in this image"
```

### Spatial Relationships

```python
query = "Find any objects on the table"
```

### Verification

```python
query = "Is there a refrigerator in this kitchen?"
```

### Combined Analysis

```python
query = "Analyze this street scene. Count people and vehicles, identify their locations relative to each other, and note any distinctive colors."
```

## Advanced Configuration

### Custom Configuration

Langvio can be configured using YAML files:

```python
pipeline = create_pipeline(config_path="path/to/your/config.yaml")
```

Example configuration file:

```yaml
llm:
  default: "gpt-4"
  models:
    gpt-4:
      model_name: "gpt-4-turbo"
      model_kwargs:
        temperature: 0.1

vision:
  default: "yolo"
  models:
    yolo:
      type: "yolo"
      model_path: "yolov11x.pt"  # Using a larger model
      confidence: 0.3

media:
  output_dir: "./custom_output"
  visualization:
    box_color: [0, 0, 255]  # Red boxes
    text_color: [255, 255, 255]  # White text
    line_thickness: 3
```

### Command Line Interface

Langvio includes a command-line interface:

```bash
langvio --query "What objects are in this image?" --media path/to/image.jpg
```

Options:

```
--query, -q      Natural language query
--media, -m      Path to image or video file
--config, -c     Path to configuration file
--llm, -l        LLM processor to use
--vision, -v     Vision processor to use
--output, -o     Output directory
--log-level      Logging level (DEBUG, INFO, WARNING, ERROR)
--log-file       Path to log file
--list-models    List available models and exit
```

## Architecture

Langvio uses a pipeline architecture that consists of:

1. **LLM Processor**: Processes natural language queries and generates explanations
2. **Vision Processor**: Detects objects and attributes in images
3. **Media Processor**: Handles image loading and visualization

The framework is designed to be extensible, allowing new models and capabilities to be added easily.

## Example Use Cases

- **Content Analysis**: "Describe what's in this image"
- **Object Finding**: "Find all instances of dogs in this photo"
- **Attribute Analysis**: "What color are the cars in this image?"
- **Scene Understanding**: "Analyze the spatial layout of this room"
- **Visual QA**: "Is there anyone wearing a red shirt in this image?"

## Supported Models

### LLM Models

- OpenAI GPT (3.5, 4)
- Google Gemini

### Vision Models

- YOLOv11 (various sizes)

## Contributing

Contributions are welcome! Please check out our contributing guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.