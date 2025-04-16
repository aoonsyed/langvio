# Complete API Reference

This document provides a comprehensive reference for the Langvio API, covering all modules, classes, and functions.

## Table of Contents

- [Core Components](#core-components)
  - [Pipeline](#pipeline)
  - [ModelRegistry](#modelregistry)
  - [Config](#config)
- [Processor Classes](#processor-classes)
  - [BaseLLMProcessor](#basellmprocessor)
  - [BaseVisionProcessor](#basevisionprocessor)
- [LLM Processors](#llm-processors)
  - [OpenAIProcessor](#openaiprocessor)
  - [GeminiProcessor](#geminiprocessor)
- [Vision Processors](#vision-processors)
  - [YOLOProcessor](#yoloprocessor)
- [Media Processing](#media-processing)
  - [MediaProcessor](#mediaprocessor)
- [Vision Utilities](#vision-utilities)
  - [ColorDetector](#colordetector)
  - [Vision Utilities Functions](#vision-utilities-functions)
- [File Utilities](#file-utilities)
  - [File Utility Functions](#file-utility-functions)
- [Logging](#logging)
  - [Logging Functions](#logging-functions)
- [Helpers and Factory Functions](#helpers-and-factory-functions)
  - [create_pipeline](#create_pipeline)
- [Command-Line Interface](#command-line-interface)
  - [CLI Options](#cli-options)

## Core Components

### Pipeline

The `Pipeline` class is the main entry point for processing images with natural language queries.

```python
from langvio.core.pipeline import Pipeline
```

#### Methods

##### `__init__(config_path=None)`
- Initialize a pipeline with an optional configuration file
- Parameters:
  - `config_path` (optional): Path to a YAML configuration file

##### `load_config(config_path)`
- Load configuration from a file
- Parameters:
  - `config_path`: Path to a YAML configuration file

##### `set_llm_processor(processor_name)`
- Set the LLM processor
- Parameters:
  - `processor_name`: Name of the processor to use

##### `set_vision_processor(processor_name)`
- Set the vision processor
- Parameters:
  - `processor_name`: Name of the processor to use

##### `process(query, media_path)`
- Process a query on media
- Parameters:
  - `query`: Natural language query
  - `media_path`: Path to an image or video file
- Returns:
  - Dictionary with results including:
    - `query`: Original query
    - `media_path`: Path to the input media
    - `media_type`: "image" or "video"
    - `output_path`: Path to the output visualization
    - `explanation`: Text explanation from the LLM
    - `detections`: Detection results
    - `query_params`: Parsed query parameters
    - `highlighted_objects`: Objects highlighted in the visualization

##### `_get_visualization_config(query_params)`
- Get visualization configuration based on query parameters
- Parameters:
  - `query_params`: Query parameters from LLM processor
- Returns:
  - Visualization configuration parameters

##### `_create_visualization(media_path, all_detections, highlighted_objects, query_params, is_video)`
- Create visualization with highlighted objects
- Parameters:
  - `media_path`: Path to input media
  - `all_detections`: All detection results
  - `highlighted_objects`: Objects to highlight
  - `query_params`: Query parameters
  - `is_video`: Whether the media is a video
- Returns:
  - Path to the output visualization

### ModelRegistry

The `ModelRegistry` class manages registration and retrieval of processors.

```python
from langvio.core.registry import ModelRegistry
```

#### Methods

##### `__init__()`
- Initialize empty registries

##### `register_llm_processor(name, processor_class, **kwargs)`
- Register an LLM processor
- Parameters:
  - `name`: Name to register the processor under
  - `processor_class`: Processor class
  - `**kwargs`: Additional parameters for the constructor

##### `register_vision_processor(name, processor_class, **kwargs)`
- Register a vision processor
- Parameters:
  - `name`: Name to register the processor under
  - `processor_class`: Processor class
  - `**kwargs`: Additional parameters for the constructor

##### `get_llm_processor(name, **kwargs)`
- Get an instance of an LLM processor
- Parameters:
  - `name`: Name of the registered processor
  - `**kwargs`: Override parameters for the constructor
- Returns:
  - Processor instance

##### `get_vision_processor(name, **kwargs)`
- Get an instance of a vision processor
- Parameters:
  - `name`: Name of the registered processor
  - `**kwargs`: Override parameters for the constructor
- Returns:
  - Processor instance

##### `list_llm_processors()`
- List all registered LLM processors
- Returns:
  - Dictionary of processor names to processor classes

##### `list_vision_processors()`
- List all registered vision processors
- Returns:
  - Dictionary of processor names to processor classes

##### `register_from_entrypoints()`
- Load and register processors from entry points

### Config

The `Config` class manages configuration loading and access.

```python
from langvio.config import Config
```

#### Methods

##### `__init__(config_path=None)`
- Initialize configuration
- Parameters:
  - `config_path` (optional): Path to a YAML configuration file

##### `_load_default_config()`
- Load the default configuration

##### `load_config(config_path)`
- Load configuration from a file
- Parameters:
  - `config_path`: Path to a YAML configuration file

##### `_update_config(base_config, new_config)`
- Recursively update base config with new config
- Parameters:
  - `base_config`: Base configuration to update
  - `new_config`: New configuration values

##### `get_llm_config(model_name=None)`
- Get configuration for an LLM model
- Parameters:
  - `model_name` (optional): Name of the model to get config for
- Returns:
  - Model configuration dictionary

##### `get_vision_config(model_name=None)`
- Get configuration for a vision model
- Parameters:
  - `model_name` (optional): Name of the model to get config for
- Returns:
  - Model configuration dictionary

##### `get_media_config()`
- Get media processing configuration
- Returns:
  - Media configuration dictionary

##### `get_logging_config()`
- Get logging configuration
- Returns:
  - Logging configuration dictionary

##### `get_langsmith_config()`
- Get LangSmith configuration if available
- Returns:
  - LangSmith configuration dictionary

##### `save_config(config_path)`
- Save current configuration to a file
- Parameters:
  - `config_path`: Path to save the configuration

## Processor Classes

### BaseLLMProcessor

The `BaseLLMProcessor` class is the base class for all LLM processors.

```python
from langvio.llm.base import BaseLLMProcessor
```

#### Methods

##### `__init__(name, config=None)`
- Initialize LLM processor
- Parameters:
  - `name`: Processor name
  - `config` (optional): Configuration parameters

##### `initialize()`
- Initialize the processor with its configuration
- Returns:
  - `True` if initialization was successful

##### `_initialize_llm()`
- Abstract method to initialize the specific LLM implementation
- Must be implemented by subclasses

##### `_setup_prompts()`
- Set up the prompt templates with system message

##### `parse_query(query)`
- Parse a natural language query into structured parameters
- Parameters:
  - `query`: Natural language query
- Returns:
  - Dictionary with parsed parameters

##### `_ensure_parsed_fields(parsed)`
- Ensure all required fields exist in the parsed query
- Parameters:
  - `parsed`: Parsed query dictionary
- Returns:
  - Updated parsed query dictionary

##### `generate_explanation(query, detections)`
- Generate an explanation based on detection results
- Parameters:
  - `query`: Natural language query
  - `detections`: Dictionary with detection results
- Returns:
  - Text explanation

##### `get_highlighted_objects()`
- Get the objects that were highlighted in the last explanation
- Returns:
  - List of highlighted objects with frame references

##### `is_package_installed(package_name)`
- Check if a Python package is installed
- Parameters:
  - `package_name`: Name of the package to check
- Returns:
  - `True` if the package is installed

### BaseVisionProcessor

The `BaseVisionProcessor` class is the base class for all vision processors.

```python
from langvio.vision.base import BaseVisionProcessor
```

#### Methods

##### `__init__(name, config=None)`
- Initialize vision processor
- Parameters:
  - `name`: Processor name
  - `config` (optional): Configuration parameters

##### `initialize()`
- Abstract method to initialize the processor with its configuration
- Must be implemented by subclasses
- Returns:
  - `True` if initialization was successful

##### `process_image(image_path, query_params)`
- Abstract method to process an image with the vision model
- Must be implemented by subclasses
- Parameters:
  - `image_path`: Path to the input image
  - `query_params`: Parameters from the query processor
- Returns:
  - Dictionary with detection results

##### `process_video(video_path, query_params, sample_rate=5)`
- Abstract method to process a video with the vision model
- Must be implemented by subclasses
- Parameters:
  - `video_path`: Path to the input video
  - `query_params`: Parameters from the query processor
  - `sample_rate`: Process every Nth frame
- Returns:
  - Dictionary with detection results

##### `_filter_detections(detections, query_params, image_dimensions=None)`
- Enhanced filter detections method with attribute and relationship support
- Parameters:
  - `detections`: Raw detection results
  - `query_params`: Query parameters including attributes and relationships
  - `image_dimensions` (optional): Tuple of (width, height) for relative positioning
- Returns:
  - Filtered detection results

##### `_analyze_video_for_activities(frame_detections, query_params)`
- Analyze video frames to detect activities
- Parameters:
  - `frame_detections`: Dictionary mapping frame indices to detections
  - `query_params`: Query parameters
- Returns:
  - Updated frame detections with activity information

##### `_get_image_dimensions(image_path)`
- Get dimensions of an image
- Parameters:
  - `image_path`: Path to the image
- Returns:
  - Tuple of (width, height) or None if failed

##### `_enhance_detections_with_attributes(detections, image_path)`
- Enhance detections with attribute information
- Parameters:
  - `detections`: List of detection dictionaries
  - `image_path`: Path to the image
- Returns:
  - Detections with added attributes

## LLM Processors

### OpenAIProcessor

The `OpenAIProcessor` class uses OpenAI models via LangChain.

```python
from langvio.llm.openai import OpenAIProcessor
```

#### Methods

##### `__init__(name="openai", model_name="gpt-3.5-turbo", model_kwargs=None, **kwargs)`
- Initialize OpenAI processor
- Parameters:
  - `name` (optional): Processor name
  - `model_name` (optional): Name of the OpenAI model to use
  - `model_kwargs` (optional): Additional model parameters
  - `**kwargs`: Additional processor parameters

##### `_initialize_llm()`
- Initialize the OpenAI model via LangChain
- Raises:
  - `ImportError`: If the required packages are not installed
  - `Exception`: If initialization fails

### GeminiProcessor

The `GeminiProcessor` class uses Google Gemini models via LangChain.

```python
from langvio.llm.google import GeminiProcessor
```

#### Methods

##### `__init__(name="gemini", model_name="gemini-pro", model_kwargs=None, **kwargs)`
- Initialize Gemini processor
- Parameters:
  - `name` (optional): Processor name
  - `model_name` (optional): Name of the Gemini model to use
  - `model_kwargs` (optional): Additional model parameters
  - `**kwargs`: Additional processor parameters

##### `_initialize_llm()`
- Initialize the Google Gemini model via LangChain
- Raises:
  - `ImportError`: If the required packages are not installed
  - `Exception`: If initialization fails

## Vision Processors

### YOLOProcessor

The `YOLOProcessor` class uses YOLO models for object detection.

```python
from langvio.vision.yolo.detector import YOLOProcessor
```

#### Methods

##### `__init__(name="yolo", model_path="yolov11n.pt", confidence=0.25, **kwargs)`
- Initialize YOLO processor
- Parameters:
  - `name` (optional): Processor name
  - `model_path` (optional): Path to the YOLO model
  - `confidence` (optional): Confidence threshold for detections
  - `**kwargs`: Additional parameters for YOLO

##### `initialize()`
- Initialize the YOLO model
- Returns:
  - `True` if initialization was successful

##### `process_image(image_path, query_params)`
- Process an image with YOLO with enhanced detection capabilities
- Parameters:
  - `image_path`: Path to the input image
  - `query_params`: Parameters from the query processor
- Returns:
  - Dictionary with all detection results without filtering

##### `process_video(video_path, query_params, sample_rate=5)`
- Process a video with YOLO with enhanced activity and tracking detection
- Parameters:
  - `video_path`: Path to the input video
  - `query_params`: Parameters from the query processor
  - `sample_rate`: Process every Nth frame
- Returns:
  - Dictionary with all detection results without filtering

## Media Processing

### MediaProcessor

The `MediaProcessor` class handles media file visualization and processing.

```python
from langvio.media.processor import MediaProcessor
```

#### Methods

##### `__init__(config=None)`
- Initialize media processor
- Parameters:
  - `config` (optional): Configuration parameters

##### `update_config(config)`
- Update configuration parameters
- Parameters:
  - `config`: New configuration parameters

##### `is_video(file_path)`
- Check if a file is a video based on extension
- Parameters:
  - `file_path`: Path to the file
- Returns:
  - `True` if the file is a video

##### `get_output_path(input_path, suffix="_processed")`
- Generate an output path for processed media
- Parameters:
  - `input_path`: Path to the input file
  - `suffix` (optional): Suffix to add to the filename
- Returns:
  - Output path

##### `visualize_image(image_path, output_path, detections, box_color=None, text_color=None, line_thickness=None, show_attributes=None, show_confidence=None)`
- Enhanced visualization of detections on an image
- Parameters:
  - `image_path`: Path to the input image
  - `output_path`: Path to save the output image
  - `detections`: List of detection dictionaries
  - `box_color` (optional): Color for bounding boxes (BGR)
  - `text_color` (optional): Color for text (BGR)
  - `line_thickness` (optional): Thickness of bounding box lines
  - `show_attributes` (optional): Whether to display attribute information
  - `show_confidence` (optional): Whether to display confidence scores

##### `visualize_video(video_path, output_path, frame_detections, box_color=None, text_color=None, line_thickness=None, show_attributes=None, show_confidence=None)`
- Enhanced visualization of detections on a video
- Parameters:
  - `video_path`: Path to the input video
  - `output_path`: Path to save the output video
  - `frame_detections`: Dictionary mapping frame indices to detections
  - `box_color` (optional): Color for bounding boxes (BGR)
  - `text_color` (optional): Color for text (BGR)
  - `line_thickness` (optional): Thickness of bounding box lines
  - `show_attributes` (optional): Whether to display attribute information
  - `show_confidence` (optional): Whether to display confidence scores

##### `_draw_detections_on_image(image, detections, box_color=(0, 255, 0), text_color=(255, 255, 255), line_thickness=2, show_attributes=True, show_confidence=True)`
- Enhanced method to draw detections on an image with attributes and activities
- Parameters:
  - `image`: Input image as numpy array
  - `detections`: List of detection dictionaries
  - `box_color`: Color for bounding boxes (BGR)
  - `text_color`: Color for text (BGR)
  - `line_thickness`: Thickness of bounding box lines
  - `show_attributes`: Whether to display attribute information
  - `show_confidence`: Whether to display confidence scores
- Returns:
  - Image with detections drawn

##### `_draw_dashed_line(img, pt1, pt2, color, thickness=1, gap=5)`
- Draw a dashed line on an image
- Parameters:
  - `img`: Image to draw on
  - `pt1`: First point
  - `pt2`: Second point
  - `color`: Line color
  - `thickness`: Line thickness
  - `gap`: Gap between dashes

##### `_draw_detections_on_video(input_path, output_path, frame_detections, box_color=(0, 255, 0), text_color=(255, 255, 255), line_thickness=2, show_attributes=True, show_confidence=True)`
- Enhanced method to draw detections on a video with tracking visualization
- Parameters:
  - `input_path`: Path to input video
  - `output_path`: Path to save output video
  - `frame_detections`: Dictionary mapping frame indices to detections
  - `box_color`: Color for bounding boxes (BGR)
  - `text_color`: Color for text (BGR)
  - `line_thickness`: Thickness of bounding box lines
  - `show_attributes`: Whether to display attribute information
  - `show_confidence`: Whether to display confidence scores

##### `_get_color_for_id(track_id)`
- Generate a consistent color for a given track ID
- Parameters:
  - `track_id`: Track identifier
- Returns:
  - BGR color tuple

## Vision Utilities

### ColorDetector

The `ColorDetector` class provides color detection capabilities.

```python
from langvio.vision.color_detection import ColorDetector
```

#### Methods

##### `detect_color(image_region, return_all=False, threshold=0.15)`
- Detect the dominant color(s) in an image region using HSV color space
- Parameters:
  - `image_region`: Image region as numpy array (BGR format)
  - `return_all` (optional): If True, returns all detected colors with percentages
  - `threshold` (optional): Minimum percentage for a color to be considered
- Returns:
  - Dominant color name if return_all=False, or dictionary of {color_name: percentage} if return_all=True

##### `detect_colors_layered(image_region, max_colors=3)`
- Detect up to max_colors different colors in the image region in order of dominance
- Parameters:
  - `image_region`: Image region as numpy array (BGR format)
  - `max_colors`: Maximum number of colors to return
- Returns:
  - List of color names in order of dominance

##### `get_color_profile(image_region)`
- Get a comprehensive color profile of the image region
- Parameters:
  - `image_region`: Image region as numpy array (BGR format)
- Returns:
  - Dictionary with color information

##### `get_color_name(bgr_color)`
- Get the name of a color given its BGR values
- Parameters:
  - `bgr_color`: Tuple of (Blue, Green, Red) values (0-255)
- Returns:
  - Name of the closest matching color

##### `find_objects_by_color(image, target_color)`
- Create a mask highlighting areas of the specified color in the image
- Parameters:
  - `image`: Input image as numpy array (BGR format)
  - `target_color`: Color name to find
- Returns:
  - Binary mask where areas of the target color are white (255)

##### `visualize_colors(image_region)`
- Create a visualization of detected colors in the image region
- Parameters:
  - `image_region`: Image region as numpy array (BGR format)
- Returns:
  - Visualization image with color information

### Vision Utilities Functions

```python
from langvio.vision.utils import (
    extract_detections,
    calculate_relative_positions,
    detect_spatial_relationships,
    detect_activities,
    filter_by_attributes,
    filter_by_spatial_relations,
    filter_by_activities
)
```

#### Functions

##### `extract_detections(results)`
- Extract detections from YOLO results with enhanced attributes
- Parameters:
  - `results`: Raw YOLO results
- Returns:
  - List of detection dictionaries with enhanced attributes

##### `calculate_relative_positions(detections, image_width, image_height)`
- Calculate relative positions and sizes of detections
- Parameters:
  - `detections`: List of detection dictionaries
  - `image_width`: Width of the image
  - `image_height`: Height of the image
- Returns:
  - Updated list of detections with relative position information

##### `detect_spatial_relationships(detections, distance_threshold=0.2)`
- Detect spatial relationships between objects
- Parameters:
  - `detections`: List of detection dictionaries
  - `distance_threshold`: Threshold for 'near' relationship
- Returns:
  - Updated list of detections with relationship information

##### `detect_activities(frame_detections, min_frames=3)`
- Detect activities across video frames based on object positions
- Parameters:
  - `frame_detections`: Dictionary mapping frame indices to detections
  - `min_frames`: Minimum frames required to detect an activity
- Returns:
  - Updated frame detections with activity information

##### `filter_by_attributes(detections, required_attributes)`
- Filter detections by required attributes
- Parameters:
  - `detections`: List of detection dictionaries
  - `required_attributes`: List of required attribute dictionaries
- Returns:
  - Filtered list of detections

##### `filter_by_spatial_relations(detections, required_relations)`
- Filter detections by required spatial relationships
- Parameters:
  - `detections`: List of detection dictionaries
  - `required_relations`: List of required relationship dictionaries
- Returns:
  - Filtered list of detections

##### `filter_by_activities(detections, required_activities)`
- Filter detections by required activities
- Parameters:
  - `detections`: List of detection dictionaries
  - `required_activities`: List of required activities
- Returns:
  - Filtered list of detections

## File Utilities

### File Utility Functions

```python
from langvio.utils.file_utils import (
    ensure_directory,
    get_file_extension,
    is_image_file,
    is_video_file,
    create_temp_copy,
    get_files_in_directory
)
```

#### Functions

##### `ensure_directory(directory)`
- Ensure a directory exists
- Parameters:
  - `directory`: Directory path

##### `get_file_extension(file_path)`
- Get the extension of a file
- Parameters:
  - `file_path`: Path to the file
- Returns:
  - File extension (lowercase)

##### `is_image_file(file_path)`
- Check if a file is an image based on extension
- Parameters:
  - `file_path`: Path to the file
- Returns:
  - `True` if the file is an image

##### `is_video_file(file_path)`
- Check if a file is a video based on extension
- Parameters:
  - `file_path`: Path to the file
- Returns:
  - `True` if the file is a video

##### `create_temp_copy(file_path, delete=True)`
- Create a temporary copy of a file
- Parameters: