"""
Example script for verification queries with Langvio
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
    """Run a verification example"""
    # Create default pipeline
    pipeline = create_pipeline()

    # Create output directory
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Example for image verification
    image_path = "examples/data/kitchen.jpg"  # Replace with your image path

    if os.path.exists(image_path):
        print(f"\n--- Processing image for verification: {image_path} ---")

        # Simple verification query
        query = "Is there a refrigerator in this kitchen?"
        print(f"Query: {query}")

        # Process the query
        result = pipeline.process(query, image_path)

        # Display results
        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")

        # Verify manually
        has_refrigerator = any(det["label"] == "refrigerator" for det in result["detections"]["0"])
        print(f"Verification: {'Yes' if has_refrigerator else 'No'}, refrigerator was "
              f"{'detected' if has_refrigerator else 'not detected'}")

        # Complex verification query
        query = "Are there any fruits on the kitchen counter?"
        print(f"\nQuery: {query}")

        # Process the query
        result = pipeline.process(query, image_path)

        # Display results
        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")

        # Verify manually
        fruit_classes = ["apple", "banana", "orange"]
        fruits_on_counter = []

        for det in result["detections"]["0"]:
            if det["label"] in fruit_classes:
                # Check if it's on the counter
                for rel in det.get("relationships", []):
                    if (rel["object"] in ["dining table", "counter"] and
                            any(r in ["on_top_of", "on"] for r in rel["relations"])):
                        fruits_on_counter.append(det["label"])

        if fruits_on_counter:
            print(f"Verification: Yes, found {', '.join(fruits_on_counter)} on the counter")
        else:
            print("Verification: No fruits detected on the counter")

    # Example for video verification
    video_path = "examples/data/park_video.mp4"  # Replace with your video path

    if os.path.exists(video_path):
        print(f"\n--- Processing video for verification: {video_path} ---")

        # Simple verification query
        query = "Is there a dog in this video?"
        print(f"Query: {query}")

        # Process the query
        result = pipeline.process(query, video_path)

        # Display results
        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")

        # Verify manually
        has_dog = False
        dog_frames = []

        for frame_key, detections in result["detections"].items():
            if any(det["label"] == "dog" for det in detections):
                has_dog = True
                dog_frames.append(frame_key)

        if has_dog:
            print(f"Verification: Yes, dog(s) detected in {len(dog_frames)} frames")
            print(f"First appearance in frame: {min(dog_frames) if dog_frames else 'N/A'}")
        else:
            print("Verification: No dogs detected in the video")

        # Complex verification query
        query = "Is there a child playing with a ball in the park?"
        print(f"\nQuery: {query}")

        # Process the query
        result = pipeline.process(query, video_path)

        # Display results
        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")

        # Verify manually (simplified)
        child_with_ball_frames = []

        for frame_key, detections in result["detections"].items():
            has_child = False
            has_ball = False

            for det in detections:
                if det["label"] == "person" and det.get("attributes", {}).get("size") == "small":
                    has_child = True
                elif det["label"] in ["sports ball", "ball"]:
                    has_ball = True

            if has_child and has_ball:
                child_with_ball_frames.append(frame_key)

        if child_with_ball_frames:
            print(f"Verification: Yes, potential child with ball detected in {len(child_with_ball_frames)} frames")
        else:
            print("Verification: No clear instances of child playing with ball detected")


if __name__ == "__main__":
    main()