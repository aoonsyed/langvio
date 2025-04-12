"""
Example script for activity detection and tracking with Langvio
"""

import os
import logging
from langvio import create_pipeline

# Load environment variables from .env file (for API keys)

# Set up logging
logging.basicConfig(level=logging.INFO)


def main():
    """Run an activity detection and tracking example"""
    # Create default pipeline
    pipeline = create_pipeline()

    # Create output directory
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Example for video activity detection
    video_path = "examples/data/people_walking.mp4"  # Replace with your video path

    if os.path.exists(video_path):
        print(f"\n--- Processing video for activities: {video_path} ---")

        # Activity detection query
        query = "Find all people walking in this video"
        print(f"Query: {query}")

        # Process the query
        result = pipeline.process(query, video_path)

        # Display results
        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")

        # Count people with walking activity
        walking_people = {}

        for frame_key, detections in result["detections"].items():
            walking_count = 0

            for det in detections:
                if det["label"] == "person" and "walking" in det.get("activities", []):
                    walking_count += 1

            if walking_count > 0:
                walking_people[frame_key] = walking_count

        # Summary of walking people
        total_frames_with_walking = len(walking_people)
        if total_frames_with_walking > 0:
            print(f"\nPeople walking detected in {total_frames_with_walking} frames")
            avg_walking = sum(walking_people.values()) / len(walking_people)
            print(f"Average of {avg_walking:.1f} walking people per frame")
        else:
            print("\nNo walking people detected in the video")

        # More complex activity query
        query = "Track people running versus people standing still"
        print(f"\nQuery: {query}")

        # Process the query
        result = pipeline.process(query, video_path)

        # Display results
        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")

        # Analyze activities
        activity_counts = {"running": 0, "stationary": 0}

        for frame_key, detections in result["detections"].items():
            for det in detections:
                if det["label"] == "person":
                    for activity in det.get("activities", []):
                        if activity in activity_counts:
                            activity_counts[activity] += 1

        # Print activity summary
        print("\nActivity summary:")
        for activity, count in activity_counts.items():
            print(f"- {activity}: {count} instances detected")

    # Example for object tracking in video
    tracking_video_path = "examples/data/crossing_video.mp4"  # Replace with your video path

    if os.path.exists(tracking_video_path):
        print(f"\n--- Processing video for tracking: {tracking_video_path} ---")

        # Object tracking query
        query = "Track all vehicles moving through the scene"
        print(f"Query: {query}")

        # Process the query
        result = pipeline.process(query, tracking_video_path)

        # Display results
        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")

        # Analyze tracking data
        tracked_objects = {}

        for frame_key, detections in result["detections"].items():
            for det in detections:
                if det["label"] in ["car", "truck", "bus", "motorcycle"] and "track_id" in det:
                    track_id = det["track_id"]

                    if track_id not in tracked_objects:
                        tracked_objects[track_id] = {
                            "type": det["label"],
                            "frames": []
                        }

                    tracked_objects[track_id]["frames"].append(int(frame_key))

        # Print tracking summary
        print(f"\nTracked {len(tracked_objects)} vehicles through the video")

        # Look at the 3 longest tracks
        if tracked_objects:
            longest_tracks = sorted(tracked_objects.items(),
                                    key=lambda x: len(x[1]["frames"]),
                                    reverse=True)[:3]

            print("\nLongest vehicle tracks:")
            for track_id, track_info in longest_tracks:
                frame_count = len(track_info["frames"])
                start_frame = min(track_info["frames"])
                end_frame = max(track_info["frames"])

                print(f"- {track_info['type']} (ID {track_id}): visible for {frame_count} frames, "
                      f"from frame {start_frame} to {end_frame}")


if __name__ == "__main__":
    main()