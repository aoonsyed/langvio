"""
Walkthrough of Langvio's Enhanced Capabilities
"""

# Enhanced Capabilities Walkthrough

This document walks through the enhanced capabilities of langvio and explains the example scripts.

## 1. Basic Object Detection

The `basic_detection_example.py` script demonstrates the fundamental capability of object detection. It:

- Detects objects in an image or video
- Shows what classes of objects are present
- Outputs a visualization with bounding boxes

Example query: "What objects are in this image?"

This forms the foundation for all other capabilities, as it establishes the base object detection system.

## 2. Object Counting

The `counting_example.py` script demonstrates counting objects of specific types. It:

- Counts instances of specific object classes
- Works with both images and videos
- Can count multiple object types simultaneously

Example query: "Count how many people are in this image"

This capability builds on basic detection by aggregating the detections into counts.

## 3. Attribute Detection

The `attribute_detection_example.py` script demonstrates how to detect visual attributes of objects. It:

- Identifies attributes like color and size
- Finds objects with specific attributes (e.g., "red cars")
- Works across both images and videos

Example query: "Find all red objects in this image"

The attribute detection capability adds a layer of analysis on top of the raw detections, extracting properties like color and size.

## 4. Spatial Relationship Analysis

The `spatial_relationship_example.py` script shows how to analyze spatial relationships between objects. It:

- Detects relationships like "above", "below", "next to"
- Finds objects relative to other objects
- Visualizes these relationships

Example query: "Find any objects on the table"

This capability analyzes the relative positions of detected objects to understand their spatial arrangement.

## 5. Activity Detection and Tracking

The `activity_tracking_example.py` script demonstrates tracking objects and detecting activities over time. It:

- Tracks objects across video frames
- Identifies activities like walking, running, or standing
- Analyzes movement patterns

Example query: "Find all people walking in this video"

This builds on object detection by analyzing changes over time to identify activities and track specific objects.

## 6. Verification Queries

The `verification_example.py` script shows how to verify the presence of specific objects or conditions. It:

- Answers yes/no questions about image/video content
- Verifies complex conditions (e.g., "Is there a child playing with a ball?")
- Provides evidence for the verification

Example query: "Is there a refrigerator in this kitchen?"

This capability frames the detection results as answers to verification questions rather than just reporting what's there.

## 7. Combined Analysis

The `combined_analysis_example.py` script demonstrates how all these capabilities can work together. It:

- Performs comprehensive analysis of complex scenes
- Combines counting, attributes, relationships, and activities
- Provides rich explanations of the scene

Example query: "Analyze this street scene. Count people and vehicles, identify their locations relative to each other, and note any distinctive colors."

This shows how the different capabilities can be combined for a more complete analysis of visual content.

## Key Advantages of the Enhanced System

1. **Generic Architecture**: The system uses a task-based approach rather than having to code for specific use cases.

2. **Natural Language Interface**: Users can interact with the system using natural language queries.

3. **Extensible Design**: New capabilities can be added without changing the core architecture.

4. **Rich Explanations**: The system provides detailed explanations of its findings, not just raw detection data.

5. **Multi-modal Processing**: Combines language and vision models to enable natural language visual analysis.

## Example Use Cases

The enhanced system can be applied to various domains:

- **Retail Analytics**: "Count customers in different sections of the store"
- **Security Monitoring**: "Alert when someone approaches the restricted area"
- **Traffic Analysis**: "Track vehicle movement patterns at this intersection"
- **Home Automation**: "Check if the stove was left on in the kitchen"
- **Accessibility**: "Describe what objects are on the table"
- **Sports Analysis**: "Track player movements during this game footage"

These examples demonstrate how the enhanced langvio system provides a flexible, powerful tool for visual analysis through natural language.