"""
Example script for attribute detection with Langvio
"""

import os
import logging
from dotenv import load_dotenv
from langvio import create_pipeline

# Load environment variables from .env file (for API keys)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)


def main():
    """Run an attribute detection example"""
    # Create default pipeline
    pipeline = create_pipeline()

    # Create output directory
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Example for image attribute detection
    image_path = "examples/data/colorful_objects.jpg"  # Replace with your image path

    if os.path.exists(image_path):
        print(f"\n--- Processing image: {image_path} ---")

        # Color attribute query
        query = "Find all red objects in this image"
        print(f"Query: {query}")

        # Process the query
        result = pipeline.process(query, image_path)

        # Display results
        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")

        # Show detected red objects
        print("\nDetected red objects:")
        for det in result["detections"]["0"]:
            if det.get("attributes", {}).get("color") == "red":
                print(f"- {det['label']} with confidence {det['confidence']:.2f}")

        # Size attribute query
        query = "Find all large objects in this image"
        print(f"\nQuery: {query}")

        # Process the query
        result = pipeline.process(query, image_path)

        # Display results
        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")

        # Show detected large objects
        print("\nDetected large objects:")
        for det in result["detections"]["0"]:
            if det.get("attributes", {}).get("size") == "large":
                print(f"- {det['label']} with confidence {det['confidence']:.2f}")

    # Example for video attribute detection
    video_path = "examples/data/traffic_video.mp4"  # Replace with your video path

    if os.path.exists(video_path):
        print(f"\n--- Processing video: {video_path} ---")

        # Color attribute query for video
        query = "Find all blue vehicles in this video"
        print(f"Query: {query}")

        # Process the query
        result = pipeline.process(query, video_path)

        # Display results
        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")

        # Analyze color attributes across frames
        blue_vehicles = {}

        for frame_key, detections in result["detections"].items():
            blue_in_frame = []

            for det in detections:
                if (det["label"] in ["car", "truck", "bus"] and
                        det.get("attributes", {}).get("color") == "blue"):
                    blue_in_frame.append(det["label"])

            if blue_in_frame:
                blue_vehicles[frame_key] = blue_in_frame

        # Summary of blue vehicles
        total_frames_with_blue = len(blue_vehicles)
        if total_frames_with_blue > 0:
            print(f"\nBlue vehicles detected in {total_frames_with_blue} frames")
            print("Example frames with blue vehicles:")

            # Show a few example frames
            for frame_key in list(blue_vehicles.keys())[:3]:
                print(f"- Frame {frame_key}: {', '.join(blue_vehicles[frame_key])}")
        else:
            print("\nNo blue vehicles detected in the video")


if __name__ == "__main__":
    main()