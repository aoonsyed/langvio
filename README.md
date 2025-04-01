# Langvio

Connect language models to vision models for natural language visual analysis.

## Overview

Langvio is a Python library that connects Large Language Models (LLMs) with computer vision models to enable natural language querying of images and videos. It currently focuses on YOLO-based object detection but is designed to be extended to support other vision models.

## Features

- Query images and videos using natural language
- Integrate multiple language models (OpenAI, etc. via LangChain)
- Support for YOLO object detection models
- Annotate images and videos with detection results
- Explanation generation for results
- Configurable via YAML files
- Command-line interface

## Installation

```bash
# Install from PyPI
pip install langvio

# Install with all dependencies
pip install langvio[openai,dev]

# Install from source
git clone https://github.com/yourusername/langvio.git
cd langvio
pip install -e .
```

## Quick Start

```python
import langvio

# Create a pipeline with default settings
pipeline = langvio.create_pipeline()

# Process a query on an image
result = pipeline.process(
    query="Find all people in this image",
    media_path="path/to/image.jpg"
)

# Print the explanation
print(result["explanation"])

# The annotated image is saved to result["output_path"]
```

## Command-line Usage

```bash
# Process an image with a query
langvio --query "Count the number of cars" --media path/to/image.jpg

# Use a custom configuration
langvio --query "Track people walking" --media path/to/video.mp4 --config my_config.yaml

# List available models
langvio --list-models
```

## Configuration

Langvio can be configured via YAML files. Here's an example:

```yaml
llm:
  default: "langchain_openai"
  models:
    langchain_openai:
      type: "langchain"
      model_name: "gpt-3.5-turbo"
      temperature: 0.0

vision:
  default: "yolo"
  models:
    yolo:
      type: "yolo"
      model_path: "yolov8n.pt"
      confidence: 0.25

media:
  output_dir: "./output"
  visualization:
    box_color: [0, 255, 0]  # Green
    text_color: [255, 255, 255]  # White
```

## Example Queries

- "Identify humans in the video"
- "Find number of cars in the video"
- "Find emotions in the video"
- "Check if a weapon was there in the video"
- "Find if theft is occurring"
- "Find the medical condition of a patient in the video"

## Extending Langvio

Langvio is designed to be extensible. You can add support for:

- New language models by implementing `BaseLLMProcessor`
- New vision models by implementing `BaseVisionProcessor`

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.