"""
Simple example of using langvio with YOLO
"""

import os
import logging
from langvio import create_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)


def main():
    """Run a simple example"""
    # Create pipeline with default settings
    pipeline = create_pipeline()

    # Set output directory
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Path to sample media (replace with your own)
    image_path = "sample_image.jpg"
    video_path = "sample_video.mp4"

    # Example queries
    image_queries = [
        "Identify all people in this image",
        "Count the number of cars in this image",
        "Find any animals in this image"
    ]

    video_queries = [
        "Track people walking in this video",
        "Count vehicles in the video",
        "Detect any unusual activities in this video"
    ]

    # Process image queries
    if os.path.exists(image_path):
        print(f"\nProcessing image: {image_path}")

        for query in image_queries:
            print(f"\nQuery: {query}")
            result = pipeline.process(query, image_path)
            print(f"Output saved to: {result['output_path']}")
            print(f"Explanation: {result['explanation']}")

    # Process video queries
    if os.path.exists(video_path):
        print(f"\nProcessing video: {video_path}")

        for query in video_queries:
            print(f"\nQuery: {query}")
            result = pipeline.process(query, video_path)
            print(f"Output saved to: {result['output_path']}")
            print(f"Explanation: {result['explanation']}")


if __name__ == "__main__":
    main()