# Configuration Guide

Langvio uses a flexible YAML-based configuration system that allows you to customize its behavior. This guide explains how to use and customize the configuration.

## Configuration File Format

Langvio configuration files use YAML format and have the following structure:

```yaml
llm:
  default: "default_llm_name"
  models:
    model_name1:
      model_name: "actual_model_name"
      model_kwargs:
        param1: value1
        param2: value2
    model_name2:
      # Another model configuration

vision:
  default: "default_vision_name"
  models:
    model_name1:
      type: "model_type"
      model_path: "path/to/model"
      confidence: 0.25
    model_name2:
      # Another model configuration

media:
  output_dir: "./output"
  temp_dir: "./temp"
  visualization:
    box_color: [0, 255, 0]  # BGR format
    text_color: [255, 255, 255]
    line_thickness: 2
    show_attributes: true
    show_confidence: true

logging:
  level: "INFO"
  file: "langvio.log"
```

## Default Configuration

Langvio comes with a default configuration that is used if no custom configuration is provided:

```yaml
llm:
  default: "gemini"
  models:
    gemini:
      model_name: "gemini-2.0-flash"
      model_kwargs:
        temperature: 0.2
        max_tokens: 1024
    gpt-3:
      model_name: "gpt-3.5-turbo"
      model_kwargs:
        temperature: 0.0
        max_tokens: 1024
    gpt-4:
      model_name: "gpt-4-turbo"
      model_kwargs:
        temperature: 0.1
        max_tokens: 2048

vision:
  default: "yolo"
  models:
    yolo:
      type: "yolo"
      model_path: "yolo11n.pt"  # Default: smallest/fastest model
      confidence: 0.25
    yolo_medium:
      type: "yolo"
      model_path: "yolo11m.pt"  # Medium model - balanced
      confidence: 0.25
    yolo_large:
      type: "yolo"
      model_path: "yolo11x.pt"  # Large model - most accurate
      confidence: 0.25

media:
  output_dir: "./output"
  temp_dir: "./temp"
  visualization:
    box_color: [0, 255, 0]  # Green boxes
    text_color: [255, 255, 255]  # White text
    line_thickness: 2
    show_attributes: true
    show_confidence: true

logging:
  level: "INFO"
  file: "langvio.log"
```

## Loading Configuration

You can load a custom configuration when creating a pipeline:

```python
from langvio import create_pipeline

# Create a pipeline with a custom configuration
pipeline = create_pipeline(config_path="path/to/your/config.yaml")
```

Or load it directly with the `Config` class:

```python
from langvio.config import Config

# Load configuration
config = Config("path/to/your/config.yaml")
```

## Configuration Sections

### LLM Configuration

The `llm` section configures the language model processors:

```yaml
llm:
  default: "gpt-4"  # Default LLM to use
  models:
    gpt-4:
      model_name: "gpt-4-turbo"  # Actual model name to use with the API
      model_kwargs:  # Parameters passed to the model
        temperature: 0.1
        max_tokens: 2048
    gpt-3:
      model_name: "gpt-3.5-turbo"
      model_kwargs:
        temperature: 0.0
        max_tokens: 1024
    gemini:
      model_name: "gemini-2.0-flash"
      model_kwargs:
        temperature: 0.2
        max_tokens: 1024
```

Key parameters:
- `default`: Name of the default LLM processor to use
- `models`: Dictionary of model configurations
  - `model_name`: Specific model name used with the API
  - `model_kwargs`: Parameters passed to the model API
    - `temperature`: Controls randomness (0.0-1.0)
    - `max_tokens`: Maximum number of tokens in the response

### Vision Configuration

The `vision` section configures the vision processors:

```yaml
vision:
  default: "yolo"  # Default vision processor to use
  models:
    yolo:
      type: "yolo"
      model_path: "yolov11n.pt"  # Path to model file or name of a standard model
      confidence: 0.25  # Detection confidence threshold
    yolo_medium:
      type: "yolo"
      model_path: "yolov11m.pt"
      confidence: 0.25
    yolo_large:
      type: "yolo"
      model_path: "yolov11x.pt"
      confidence: 0.25
```

Key parameters:
- `default`: Name of the default vision processor to use
- `models`: Dictionary of model configurations
  - `type`: Type of the vision model
  - `model_path`: Path to model file or name of a standard model
  - `confidence`: Detection confidence threshold (0.0-1.0)

### Media Configuration

The `media` section configures media processing and visualization:

```yaml
media:
  output_dir: "./output"  # Directory for output files
  temp_dir: "./temp"  # Directory for temporary files
  visualization:
    box_color: [0, 255, 0]  # Color for bounding boxes (BGR format)
    text_color: [255, 255, 255]  # Color for text
    line_thickness: 2  # Thickness of bounding box lines
    show_attributes: true  # Whether to show object attributes
    show_confidence: true  # Whether to show confidence scores
```

Key parameters:
- `output_dir`: Directory for output files
- `temp_dir`: Directory for temporary files
- `visualization`: Visualization settings
  - `box_color`: Color for bounding boxes in BGR format [B, G, R]
  - `text_color`: Color for text in BGR format
  - `line_thickness`: Thickness of bounding box lines
  - `show_attributes`: Whether to show object attributes
  - `show_confidence`: Whether to show confidence scores

### Logging Configuration

The `logging` section configures logging:

```yaml
logging:
  level: "INFO"  # Logging level
  file: "langvio.log"  # Log file (set to null for console only)
```

Key parameters:
- `level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `file`: Path to log file (set to null for console only)

## Examples

### Minimal Configuration

A minimal configuration might only specify the default models:

```yaml
llm:
  default: "gpt-3"

vision:
  default: "yolo"
```

### Custom Visualization

Customize the visualization appearance:

```yaml
media:
  output_dir: "./my_results"
  visualization:
    box_color: [0, 0, 255]  # Red boxes (BGR format)
    text_color: [255, 255, 0]  # Cyan text
    line_thickness: 3
    show_attributes: true
    show_confidence: false
```

### Multi-model Configuration

Configure multiple models for different use cases:

```yaml
llm:
  default: "gpt-3"
  models:
    gpt-3:
      model_name: "gpt-3.5-turbo"
      model_kwargs:
        temperature: 0.0
    gpt-4:
      model_name: "gpt-4-turbo"
      model_kwargs:
        temperature: 0.1
    gemini:
      model_name: "gemini-2.0-flash"
      model_kwargs:
        temperature: 0.2

vision:
  default: "yolo_fast"
  models:
    yolo_fast:
      type: "yolo"
      model_path: "yolov11n.pt"
      confidence: 0.25
    yolo_accurate:
      type: "yolo"
      model_path: "yolov11x.pt"
      confidence: 0.3
```

## Programmatic Configuration

You can also modify the configuration programmatically:

```python
from langvio import create_pipeline
from langvio.config import Config

# Load config
config = Config("path/to/config.yaml")

# Modify config
config.config["media"]["visualization"]["box_color"] = [255, 0, 0]  # Blue boxes

# Save modified config
config.save_config("path/to/modified_config.yaml")

# Create pipeline with modified config
pipeline = create_pipeline(config_path="path/to/modified_config.yaml")
```

## Environment Variables for API Keys

Langvio uses environment variables for API keys. You should set these in your environment or use a `.env` file:

```
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
```

## Configuration Best Practices

1. **Model Selection**: Choose models based on your needs:
   - `yolov11n.pt`: Fastest, but less accurate
   - `yolov11m.pt`: Balanced speed and accuracy
   - `yolov11x.pt`: Most accurate, but slower

2. **Confidence Threshold**: Adjust based on your requirements:
   - Lower values (e.g., 0.2): More detections, but potentially more false positives
   - Higher values (e.g., 0.4): Fewer detections, but more confidence in each one

3. **Temperature**: Adjust based on the creativity needed:
   - Lower values (e.g., 0.0): More deterministic, focused responses
   - Higher values (e.g., 0.7): More creative, varied responses

4. **Visualization**: Customize for your use case:
   - For presentations: Increase line thickness and use distinctive colors
   - For automated processing: You might disable visualization entirely

5. **Logging**: Configure based on your debugging needs:
   - `DEBUG`: Verbose logging for development
   - `INFO`: General operational info
   - `WARNING`: Only issues and warnings for production