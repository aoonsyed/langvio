"""
Example script for spatial relationship detection with Langvio
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
    """Run a spatial relationship detection example"""
    # Create default pipeline
    pipeline = create_pipeline()

    # Create output directory
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Example for image spatial relationships
    image_path = "examples/data/living_room.jpg"  # Replace with your image path

    if os.path.exists(image_path):
        print(f"\n--- Processing image: {image_path} ---")

        # Spatial relationship query - "on"
        query = "Find any objects on the table"
        print(f"Query: {query}")

        # Process the query
        result = pipeline.process(query, image_path)

        # Display results
        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")

        # Show detected spatial relationships
        print("\nDetected objects on table:")
        for det in result["detections"]["0"]:
            for rel in det.get("relationships", []):
                if rel["object"] == "dining table" and "on_top_of" in rel["relations"]:
                    print(f"- {det['label']} with confidence {det['confidence']:.2f}")

        # Spatial relationship query - "near"
        query = "What objects are near the couch?"
        print(f"\nQuery: {query}")

        # Process the query
        result = pipeline.process(query, image_path)

        # Display results
        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")

        # Show detected spatial relationships
        print("\nDetected objects near couch:")
        for det in result["detections"]["0"]:
            for rel in det.get("relationships", []):
                if rel["object"] == "couch" and "near" in rel["relations"]:
                    print(f"- {det['label']} with confidence {det['confidence']:.2f}")

    # Example for video spatial relationships
    video_path = "examples/data/street_scene_video.mp4"  # Replace with your video path

    if os.path.exists(video_path):
        print(f"\n--- Processing video: {video_path} ---")

        # Spatial relationship in video
        query = "Find pedestrians near cars"
        print(f"Query: {query}")

        # Process the query
        result = pipeline.process(query, video_path)

        # Display results
        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")

        # Analyze spatial relationships across frames
        people_near_cars = {}

        for frame_key, detections in result["detections"].items():
            for det in detections:
                if det["label"] == "person":
                    for rel in det.get("relationships", []):
                        if rel["object"] == "car" and "near" in rel["relations"]:
                            if frame_key not in people_near_cars:
                                people_near_cars[frame_key] = 0
                            people_near_cars[frame_key] += 1

        # Summary of people near cars
        total_frames_with_people_near_cars = len(people_near_cars)
        if total_frames_with_people_near_cars > 0:
            print(f"\nPeople near cars detected in {total_frames_with_people_near_cars} frames")
            print("Example frames with people near cars:")

            # Show a few example frames
            for frame_key in list(people_near_cars.keys())[:3]:
                print(f"- Frame {frame_key}: {people_near_cars[frame_key]} people near cars")

            # Calculate the frame with most people near cars
            max_frame = max(people_near_cars.items(), key=lambda x: x[1])
            print(f"\nFrame {max_frame[0]} has the most people near cars: {max_frame[1]}")
        else:
            print("\nNo people near cars detected in the video")


if __name__ == "__main__":
    main()