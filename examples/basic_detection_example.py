"""
Example script for basic object detection with Langvio
"""

import os
import logging
from langvio import create_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)


def main():
    """Run a basic object detection example"""
    # Create default pipeline
    pipeline = create_pipeline()

    # Create output directory
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Example for image detection
    image_path = "data/sample_image.jpeg"  # Replace with your image path

    if os.path.exists(image_path):
        print(f"\n--- Processing image: {image_path} ---")

        # Basic detection query
        query = "What objects are in this image?"
        print(f"Query: {query}")

        # Process the query
        result = pipeline.process(query, image_path)

        # Display results
        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")
        print("\nDetected objects:")

        # List all detected objects
        detected_classes = set()
        for detection in result["detections"]["0"]:
            detected_classes.add(detection["label"])

        for cls in detected_classes:
            print(f"- {cls}")

    # Example for video detection
    video_path = "data/sample_video.mp4"  # Replace with your video path

    if os.path.exists(video_path):
        print(f"\n--- Processing video: {video_path} ---")

        # Basic detection query
        query = "What objects appear in this video?"
        print(f"Query: {query}")

        # Process the query
        result = pipeline.process(query, video_path)

        # Display results
        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")
        print("\nDetected objects:")

        # Aggregate all detected objects across frames
        detected_classes = set()
        for frame_key, detections in result["detections"].items():
            for detection in detections:
                detected_classes.add(detection["label"])

        for cls in detected_classes:
            print(f"- {cls}")


if __name__ == "__main__":
    main()