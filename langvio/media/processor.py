"""
Media processing utilities
"""

import os
import logging
from typing import Dict, Any, List, Optional

import cv2
import numpy as np

from langvio.media.visualization import draw_detections_on_image, draw_detections_on_video


class MediaProcessor:
    """Processor for handling media files (images and videos)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize media processor.

        Args:
            config: Configuration parameters
        """
        self.config = config or {
            "output_dir": "./output",
            "temp_dir": "./temp",
            "visualization": {
                "box_color": [0, 255, 0],
                "text_color": [255, 255, 255],
                "line_thickness": 2
            }
        }

        self.logger = logging.getLogger(__name__)

        # Create output and temp directories
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs(self.config["temp_dir"], exist_ok=True)

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update configuration parameters.

        Args:
            config: New configuration parameters
        """
        self.config.update(config)

        # Ensure directories exist
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs(self.config["temp_dir"], exist_ok=True)

    def is_video(self, file_path: str) -> bool:
        """
        Check if a file is a video based on extension.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is a video
        """
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        _, ext = os.path.splitext(file_path.lower())
        return ext in video_extensions

    def get_output_path(self, input_path: str, suffix: str = "_processed") -> str:
        """
        Generate an output path for processed media.

        Args:
            input_path: Path to the input file
            suffix: Suffix to add to the filename

        Returns:
            Output path
        """
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}{suffix}{ext}"

        return os.path.join(self.config["output_dir"], output_filename)

    def visualize_image(self, image_path: str, output_path: str,
                        detections: List[Dict[str, Any]]) -> None:
        """
        Visualize detections on an image.

        Args:
            image_path: Path to the input image
            output_path: Path to save the output image
            detections: List of detection dictionaries
        """
        self.logger.info(f"Visualizing {len(detections)} detections on image: {image_path}")

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Draw detections
            viz_config = self.config["visualization"]

            image_with_detections = draw_detections_on_image(
                image,
                detections,
                box_color=viz_config["box_color"],
                text_color=viz_config["text_color"],
                line_thickness=viz_config["line_thickness"]
            )

            # Save output
            cv2.imwrite(output_path, image_with_detections)
            self.logger.info(f"Saved visualized image to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error visualizing image: {e}")

    def visualize_video(self, video_path: str, output_path: str,
                        frame_detections: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Visualize detections on a video.

        Args:
            video_path: Path to the input video
            output_path: Path to save the output video
            frame_detections: Dictionary mapping frame indices to detections
        """
        self.logger.info(f"Visualizing detections on video: {video_path}")

        try:
            # Get visualization config
            viz_config = self.config["visualization"]

            # Draw detections on video
            draw_detections_on_video(
                video_path,
                output_path,
                frame_detections,
                box_color=viz_config["box_color"],
                text_color=viz_config["text_color"],
                line_thickness=viz_config["line_thickness"]
            )

            self.logger.info(f"Saved visualized video to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error visualizing video: {e}")