"""
Simple example of using langvio with Gemini to detect humans
"""

import os
import logging
from langvio import create_pipeline
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)


def main():
    """Run a simple example"""
    # Create pipeline with Gemini LLM (will exit if not available)
    pipeline = create_pipeline(llm_name="gemini")

    # Set output directory
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Path to sample media (replace with your own)
    image_path = "data/sample_image.jpeg"
    video_path = "data/sample_video.mp4"

    # Process image if it exists
    if os.path.exists(image_path):
        print(f"\nProcessing image: {image_path}")

        query = "Detect and count all people in this image"
        print(f"Query: {query}")

        result = pipeline.process(query, image_path)

        print(f"Output saved to: {result['output_path']}")
        print(f"Explanation: {result['explanation']}")

    # # Process video if it exists
    # if os.path.exists(video_path):
    #     print(f"\nProcessing video: {video_path}")
    #
    #     query = "Track all people in this video"
    #     print(f"Query: {query}")
    #
    #     result = pipeline.process(query, video_path)
    #
    #     print(f"Output saved to: {result['output_path']}")
    #     print(f"Explanation: {result['explanation']}")


if __name__ == "__main__":
    main()