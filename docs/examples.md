# Examples

This document provides practical examples of using Langvio for various visual analysis tasks.

## Basic Examples

### Object Detection

This example demonstrates basic object detection in an image:

```python
import os
from langvio import create_pipeline

# Create pipeline
pipeline = create_pipeline()

# Create output directory
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Process image
image_path = "data/sample_image.jpeg"
query = "What objects are in this image?"

result = pipeline.process(query, image_path)

# Display results
print(f"Explanation: {result['explanation']}")
print(f"Output saved to: {result['output_path']}")

# List all detected objects
detected_classes = set()
for detection in result["detections"]["0"]:
    detected_classes.add(detection["label"])

print("\nDetected objects:")
for cls in detected_classes:
    print(f"- {cls}")
```

### Object Counting

This example shows how to count specific objects in an image:

```python
import os
from langvio import create_pipeline

# Create pipeline
pipeline = create_pipeline()

# Create output directory
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Process image
image_path = "data/sample_image.jpeg"
query = "Count how many people are in this image"

result = pipeline.process(query, image_path)

# Display results
print(f"Explanation: {result['explanation']}")
print(f"Output saved to: {result['output_path']}")

# Verify the count manually
person_count = sum(1 for det in result["detections"]["0"] if det["label"] == "person")
print(f"Verified count: {person_count} people detected")
```

### Attribute Detection

This example demonstrates detecting objects with specific attributes:

```python
import os
from langvio import create_pipeline

# Create pipeline
pipeline = create_pipeline()

# Create output directory
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Process image
image_path = "data/sample_image.jpeg"
query = "Find all red objects in this image"

result = pipeline.process(query, image_path)

# Display results
print(f"Explanation: {result['explanation']}")
print(f"Output saved to: {result['output_path']}")

# Show detected red objects
print("\nDetected red objects:")
for det in result["detections"]["0"]:
    if det.get("attributes", {}).get("color") == "red":
        print(f"- {det['label']} with confidence {det['confidence']:.2f}")
```

### Combined Analysis

This example shows a comprehensive analysis combining multiple capabilities:

```python
import os
from langvio import create_pipeline

# Create pipeline
pipeline = create_pipeline()

# Create output directory
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Process image
image_path = "data/sample_image.jpeg"
query = "Analyze this street scene. Count people and vehicles, identify their locations relative to each other, and note any distinctive colors."

result = pipeline.process(query, image_path)

# Display results
print(f"Explanation: {result['explanation']}")
print(f"Output saved to: {result['output_path']}")
```

## Advanced Examples

### Using Different LLM Models

This example shows how to use a specific LLM model:

```python
import os
from langvio import create_pipeline

# Create pipeline with a specific LLM
pipeline = create_pipeline(llm_name="gpt-4")

# Create output directory
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Process image
image_path = "data/sample_image.jpeg"
query = "Provide a detailed description of this image"

result = pipeline.process(query, image_path)

# Display results
print(f"Explanation: {result['explanation']}")
print(f"Output saved to: {result['output_path']}")
```

### Using a Custom Configuration

This example demonstrates using a custom configuration:

```python
import os
from langvio import create_pipeline

# Create a pipeline with custom configuration
pipeline = create_pipeline(config_path="examples/config_examples/custom_model_config.yaml")

# Create output directory
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Process image
image_path = "data/sample_image.jpeg"
query = "What objects are in this image?"

result = pipeline.process(query, image_path)

# Display results
print(f"Explanation: {result['explanation']}")
print(f"Output saved to: {result['output_path']}")
```

Example custom configuration file:

```yaml
# Custom langvio configuration

llm:
  default: "langchain_openai"
  models:
    langchain_openai:
      type: "langchain"
      model_name: "gpt-4-turbo"
      temperature: 0.1

vision:
  default: "yolo"
  models:
    yolo:
      type: "yolo"
      model_path: "yolov8x.pt"  # Using a larger model
      confidence: 0.3

media:
  output_dir: "./custom_output"
  visualization:
    box_color: [0, 0, 255]  # Red boxes
    text_color: [255, 255, 255]  # White text
    line_thickness: 3
```

### Accessing Detection Details

This example shows how to access and use detailed detection information:

```python
import os
from langvio import create_pipeline

# Create pipeline
pipeline = create_pipeline()

# Create output directory
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Process image
image_path = "data/sample_image.jpeg"
query = "What objects are in this image?"

result = pipeline.process(query, image_path)

# Access detailed detection information
detections = result["detections"]["0"]  # For images, frame key is "0"

# Group detections by label
detections_by_label = {}
for det in detections:
    label = det["label"]
    if label not in detections_by_label:
        detections_by_label[label] = []
    detections_by_label[label].append(det)

# Print detection statistics
print("\nDetection statistics:")
for label, dets in detections_by_label.items():
    print(f"{label}: {len(dets)} instances")
    
    # Calculate average confidence
    avg_confidence = sum(d["confidence"] for d in dets) / len(dets)
    print(f"  Average confidence: {avg_confidence:.2f}")
    
    # Print attribute information if available
    for det in dets:
        if "attributes" in det:
            attr_str = ", ".join(f"{k}: {v}" for k, v in det["attributes"].items())
            print(f"  Instance with attributes: {attr_str}")
```

### Finding Objects with Spatial Relationships

This example demonstrates finding objects based on spatial relationships:

```python
import os
from langvio import create_pipeline

# Create pipeline
pipeline = create_pipeline()

# Create output directory
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Process image
image_path = "data/sample_image.jpeg"
query = "Find objects that are on top of tables or desks"

result = pipeline.process(query, image_path)

# Display results
print(f"Explanation: {result['explanation']}")
print(f"Output saved to: {result['output_path']}")

# Manually check spatial relationships
print("\nDetected spatial relationships:")
for det in result["detections"]["0"]:
    if "relationships" in det:
        for rel in det["relationships"]:
            if "relations" in rel and "object" in rel:
                rel_obj = rel["object"]
                for relation in rel["relations"]:
                    print(f"- {det['label']} is {relation} {rel_obj}")
```

## Using the Command Line Interface

Langvio includes a command-line interface for quick analysis without writing code:

```bash
# Basic usage
langvio --query "What objects are in this image?" --media path/to/image.jpg

# With custom config
langvio --query "Count how many people are in this image" --media path/to/image.jpg --config path/to/config.yaml

# Specifying output directory
langvio --query "Find all red objects" --media path/to/image.jpg --output ./results

# Using a specific LLM
langvio --query "Describe this image in detail" --media path/to/image.jpg --llm gpt-4

# List available models
langvio --list-models
```

## Real-World Use Cases

### Content Moderation

This example shows how to use Langvio for basic content moderation:

```python
import os
from langvio import create_pipeline

def check_image_content(image_path):
    """Check if an image contains potentially sensitive content."""
    # Create pipeline
    pipeline = create_pipeline()
    
    # Query for potentially sensitive content
    query = "Does this image contain any inappropriate, violent, or adult content? Please be specific."
    
    # Process image
    result = pipeline.process(query, image_path)
    
    # Return the explanation
    return {
        "explanation": result["explanation"],
        "output_path": result["output_path"],
        "detections": result["detections"]
    }

# Test with an image
image_path = "data/sample_image.jpeg"
result = check_image_content(image_path)
print(f"Content analysis: {result['explanation']}")
```

### Retail Product Analysis

This example demonstrates analyzing retail product images:

```python
import os
from langvio import create_pipeline

def analyze_product_image(image_path):
    """Analyze a retail product image."""
    # Create pipeline
    pipeline = create_pipeline(llm_name="gpt-4")  # Using a more capable LLM
    
    # Query for product analysis
    query = "Analyze this product image. Identify the type of product, brand if visible, approximate size, color, and any distinctive features or text visible on the packaging."
    
    # Process image
    result = pipeline.process(query, image_path)
    
    # Return the analysis
    return {
        "explanation": result["explanation"],
        "output_path": result["output_path"],
        "detections": result["detections"]
    }

# Test with a product image
image_path = "data/product_image.jpg"
if os.path.exists(image_path):
    result = analyze_product_image(image_path)
    print(f"Product analysis: {result['explanation']}")
```

### Social Media Content Analysis

This example shows how to analyze the content of social media images:

```python
import os
from langvio import create_pipeline

def analyze_social_media_image(image_path):
    """Analyze a social media image for content and context."""
    # Create pipeline
    pipeline = create_pipeline()
    
    # Comprehensive query for social media analysis
    query = "Analyze this social media image. What is the main subject? Are there people, and if so, how many and what are they doing? Is this in a recognizable location or setting? Are there any text overlays or captions visible? Does the image appear to be promoting something? Describe the overall mood or tone of the image."
    
    # Process image
    result = pipeline.process(query, image_path)
    
    # Return the analysis
    return {
        "explanation": result["explanation"],
        "output_path": result["output_path"],
        "objects": [det["label"] for det in result["detections"]["0"]]
    }

# Test with a social media image
image_path = "data/social_media_image.jpg"
if os.path.exists(image_path):
    result = analyze_social_media_image(image_path)
    print(f"Social media content analysis: {result['explanation']}")
    print(f"Detected objects: {', '.join(result['objects'])}")
```

## Tips and Best Practices

### Crafting Effective Queries

The way you phrase your query can significantly impact results:

#### General vs. Specific Queries

```python
# General query (less effective)
query = "What's in this image?"

# Specific query (more effective)
query = "Identify all people, vehicles, and animals in this image, including their positions relative to each other."
```

#### Task-Focused Queries

```python
# Counting task
query = "Count how many people are in this image, and specify if any are children."

# Verification task
query = "Is there anyone wearing a red shirt in this image?"

# Analysis task
query = "Analyze the spatial arrangement of furniture in this room, noting any unusual placements."
```

### Handling Large Images

For large or high-resolution images, you may want to resize them first:

```python
import cv2
import os
from langvio import create_pipeline

def process_large_image(image_path, max_width=1280, max_height=1280):
    """Process a large image by resizing it first."""
    # Create output directory
    os.makedirs("./temp", exist_ok=True)
    
    # Load and resize image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Calculate new dimensions
    if w > max_width or h > max_height:
        if w/max_width > h/max_height:
            new_w = max_width
            new_h = int(h * (max_width/w))
        else:
            new_h = max_height
            new_w = int(w * (max_height/h))
        
        # Resize image
        img = cv2.resize(img, (new_w, new_h))
        
        # Save resized image
        temp_path = "./temp/resized_image.jpg"
        cv2.imwrite(temp_path, img)
    else:
        temp_path = image_path
    
    # Process the resized image
    pipeline = create_pipeline()
    result = pipeline.process("What objects are in this image?", temp_path)
    
    return result

# Test with a large image
image_path = "data/large_image.jpg"
if os.path.exists(image_path):
    result = process_large_image(image_path)
    print(f"Analysis of resized image: {result['explanation']}")
```

### Enhancing Specific Detection Types

For specific detection tasks, you can create specialized pipelines:

```python
import os
from langvio import create_pipeline

def create_specialized_pipeline(task_type):
    """Create a specialized pipeline for a specific detection task."""
    # Create custom config for this task
    if task_type == "people":
        # For people detection, prioritize accuracy
        pipeline = create_pipeline(vision_name="yolo_large")
    elif task_type == "vehicles":
        # For vehicle detection, use a balanced approach
        pipeline = create_pipeline(vision_name="yolo_medium")
    elif task_type == "text":
        # For text detection, use a more detailed LLM
        pipeline = create_pipeline(llm_name="gpt-4")
    else:
        # Default pipeline
        pipeline = create_pipeline()
    
    return pipeline

# Test specialized pipelines
image_path = "data/sample_image.jpeg"
if os.path.exists(image_path):
    # People detection
    people_pipeline = create_specialized_pipeline("people")
    people_result = people_pipeline.process("Count all people in this image", image_path)
    print(f"People detection: {people_result['explanation']}")
    
    # Vehicle detection
    vehicle_pipeline = create_specialized_pipeline("vehicles")
    vehicle_result = vehicle_pipeline.process("Count and identify all vehicles in this image", image_path)
    print(f"Vehicle detection: {vehicle_result['explanation']}")
```