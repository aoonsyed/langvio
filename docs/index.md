# Langvio Documentation

Welcome to the Langvio documentation. Langvio is a Python framework that connects language models with vision models to enable natural language visual analysis.

## What is Langvio?

Langvio is a framework that allows you to analyze images (and videos) using natural language queries. It connects large language models (LLMs) like OpenAI's GPT or Google's Gemini with computer vision models like YOLO to provide rich, natural language analysis of visual content.

With Langvio, you can:

- Detect objects in images using natural language queries
- Count specific types of objects
- Find objects with specific attributes (colors, sizes, etc.)
- Analyze spatial relationships between objects
- Track objects across video frames
- Detect activities in videos
- Get natural language explanations of visual content

## Key Features

- **Natural Language Interface**: Query visual content using plain English
- **Multimodal Integration**: Seamlessly connects LLMs with vision models
- **Flexible Architecture**: Supports multiple LLM providers
- **Rich Analysis**: Detect objects, attributes, spatial relationships, and more
- **YOLO Integration**: Powered by YOLOv11 for fast and accurate object detection
- **Extensible Design**: Easy to add new models and capabilities

## Documentation Sections

- [Getting Started](getting_started.md): Quick installation and first steps
- [Installation](installation.md): Detailed installation instructions
- [Usage Guide](usage.md): How to use Langvio with examples
- [Configuration](configuration.md): How to configure Langvio
- [Examples](examples.md): Example use cases and code
- [API Reference](api_reference.md): Detailed API documentation
- [Architecture](architecture.md): Overview of Langvio's architecture

## Quick Start

### Installation

```bash
pip install langvio[openai]  # or langvio[google] for Google Gemini
```

### Basic Usage

```python
import os
from langvio import create_pipeline

# Create a pipeline
pipeline = create_pipeline()

# Process an image
result = pipeline.process(
    query="What objects are in this image?",
    media_path="path/to/image.jpg"
)

# Print the results
print(f"Explanation: {result['explanation']}")
print(f"Output saved to: {result['output_path']}")
```

### Command Line

```bash
langvio --query "What objects are in this image?" --media path/to/image.jpg
```

## Requirements

- Python 3.8+
- An API key for at least one LLM provider (OpenAI, Google, etc.)
- Sufficient disk space for YOLO models (100-200MB per model)

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.