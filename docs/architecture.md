# Architecture

This document provides an overview of the Langvio architecture and how its components work together.

## Overview

Langvio connects language models (LLMs) with vision models through a modular, extensible architecture that enables natural language visual analysis. The system follows a pipeline pattern, where each component performs a specific task in the analysis process.

![Langvio Architecture Diagram](architecture_diagram.svg)

## Core Components

### Pipeline

The `Pipeline` class is the central orchestrator that:
1. Manages the configuration
2. Connects the LLM processor, vision processor, and media processor
3. Handles the overall processing flow

### LLM Processors

LLM processors are responsible for:
1. Parsing natural language queries into structured parameters
2. Generating explanations based on detection results
3. Selecting objects to highlight in visualizations

Langvio supports multiple LLM providers through an extensible interface, currently including:
- OpenAI (GPT-3.5, GPT-4)
- Google Gemini

### Vision Processors

Vision processors handle the actual image and video analysis:
1. Detecting objects in images and videos
2. Identifying attributes (color, size, etc.)
3. Analyzing spatial relationships between objects
4. Tracking objects across video frames
5. Detecting activities in videos

Currently, Langvio uses YOLO models for vision processing.

### Media Processors

Media processors handle media file operations:
1. Loading images and videos
2. Creating visualizations with bounding boxes and labels
3. Saving output files

### Model Registry

The model registry manages the registration and retrieval of processors, allowing the system to:
1. Dynamically discover available processors
2. Load processors based on configuration
3. Handle processor dependencies

## Processing Flow

1. **User Query**: The user provides a natural language query and an image/video path.

2. **Query Parsing**: The LLM processor parses the query into structured parameters, including:
   - Target objects to detect
   - Whether to count objects
   - The type of analysis (identification, counting, verification, etc.)
   - Attributes to check (color, size, etc.)
   - Spatial relationships to analyze
   - Activities to detect (for videos)

3. **Vision Processing**: The vision processor analyzes the image/video based on the query parameters:
   - Detects objects and their positions
   - Identifies attributes like color and size
   - Analyzes spatial relationships between objects
   - Tracks objects across video frames
   - Detects activities in videos

4. **LLM Explanation**: The LLM processor generates a natural language explanation based on the detection results.

5. **Visualization**: The media processor creates an output image/video with highlighted detections.

6. **Result Return**: The system returns the explanation, output file path, and detailed detection results.

## Configuration System

Langvio uses a flexible YAML-based configuration system that allows:
1. Setting default processors
2. Configuring model parameters
3. Customizing visualization options
4. Setting output directories and logging options

Example configuration:
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
      model_path: "yolov11x.pt"
      confidence: 0.3

media:
  output_dir: "./custom_output"
  visualization:
    box_color: [0, 0, 255]
    text_color: [255, 255, 255]
    line_thickness: 3
```

## Extensibility

Langvio is designed to be easily extended with:

### New LLM Providers

To add a new LLM provider:
1. Create a new class extending `BaseLLMProcessor`
2. Implement the required methods (`_initialize_llm()`)
3. Register the processor in the factory

### New Vision Models

To add a new vision model:
1. Create a new class extending `BaseVisionProcessor`
2. Implement the required methods (`process_image()`, `process_video()`)
3. Register the processor in the registry

### Enhanced Capabilities

Langvio can be extended with new capabilities such as:
1. New attribute detectors (e.g., emotion detection)
2. Additional spatial relationship analyzers
3. More sophisticated activity recognition
4. Custom visualizations

## Dependencies and Integration

Langvio is built on several core technologies:

1. **LangChain**: For LLM integration and structured outputs
2. **Ultralytics YOLO**: For object detection
3. **OpenCV**: For image and video processing
4. **PyTorch**: For deep learning operations
5. **LLM APIs**: OpenAI, Google, etc.

## Design Patterns

Langvio uses several key design patterns:

1. **Factory Pattern**: For creating processor instances
2. **Strategy Pattern**: For swappable LLM and vision processors
3. **Registry Pattern**: For managing available processors
4. **Pipeline Pattern**: For the overall processing flow
5. **Command Pattern**: For representing queries as structured commands

## Code Organization

```
langvio/
├── __init__.py                  # Package init, exports main components
├── cli.py                       # Command-line interface
├── config.py                    # Configuration management
├── default_config.yaml          # Default configuration
├── core/                        # Core components
│   ├── __init__.py
│   ├── base.py                  # Base classes
│   ├── registry.py              # Model registry
│   └── pipeline.py              # Main pipeline
├── llm/                         # LLM processors
│   ├── __init__.py
│   ├── base.py                  # Base LLM processor
│   ├── factory.py               # LLM factory
│   ├── openai.py                # OpenAI processor
│   └── google.py                # Google processor
├── vision/                      # Vision processors
│   ├── __init__.py
│   ├── base.py                  # Base vision processor
│   ├── utils.py                 # Vision utilities
│   ├── color_detection.py       # Color detection
│   └── yolo/                    # YOLO models
│       ├── __init__.py
│       └── detector.py          # YOLO processor
├── media/                       # Media handling
│   ├── __init__.py
│   ├── processor.py             # Media processor
│   └── visualization.py         # Visualization utilities
├── prompts/                     # LLM prompts
│   ├── __init__.py
│   ├── templates.py             # Prompt templates
│   └── constants.py             # Constants for prompts
└── utils/                       # Utility functions
    ├── __init__.py
    ├── file_utils.py            # File utilities
    ├── llm_utils.py             # LLM utilities
    ├── logging.py               # Logging setup
    └── vision_utils.py          # Vision utilities
```

## Future Architecture Directions

Langvio's architecture is designed to evolve in several key directions:

1. **Multi-modal Integration**: Adding support for audio and text inputs alongside images and videos.

2. **Plugin System**: A more formal plugin architecture for easier extension with new processors and capabilities.

3. **Model Caching**: Improved caching mechanisms for models and results to enhance performance.

4. **Distributed Processing**: Support for distributed processing of large media collections.

5. **Memory and Context**: Adding conversational memory for follow-up queries about the same media.

6. **Fine-tuned Models**: Support for fine-tuned models specialized for visual analysis tasks.