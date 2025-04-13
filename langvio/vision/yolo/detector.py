"""
Enhanced YOLO-based vision processor
"""

import os
import tempfile
import logging
from typing import Dict, Any, List, Optional, Tuple

import cv2
from ultralytics import YOLO

from langvio.vision.base import BaseVisionProcessor
from langvio.vision.utils import extract_detections
from langvio.prompts.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_VIDEO_SAMPLE_RATE
)


class YOLOProcessor(BaseVisionProcessor):
    """Enhanced vision processor using YOLO models"""

    def __init__(self, name: str = "yolo",
                 model_path: str = "yolov11n.pt",
                 confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 **kwargs):
        """
        Initialize YOLO processor.

        Args:
            name: Processor name
            model_path: Path to the YOLO model
            confidence: Confidence threshold for detections
            **kwargs: Additional parameters for YOLO
        """
        config = {
            "model_path": model_path,
            "confidence": confidence,
            **kwargs
        }
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.model = None

    def initialize(self) -> bool:
        """
        Initialize the YOLO model.

        Returns:
            True if initialization was successful
        """
        try:
            self.logger.info(f"Loading YOLO model: {self.config['model_path']}")
            self.model = YOLO(self.config["model_path"])
            return True
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            return False

    def process_image(self, image_path: str, query_params: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process an image with YOLO with enhanced detection capabilities.
        Modified to return all detections without filtering.

        Args:
            image_path: Path to the input image
            query_params: Parameters from the query processor

        Returns:
            Dictionary with all detection results without filtering
        """
        self.logger.info(f"Processing image: {image_path}")

        # Load model if not already loaded
        if not self.model:
            self.initialize()

        # Run detection
        try:
            # Get image dimensions for relative positioning
            image_dimensions = self._get_image_dimensions(image_path)

            # Run basic object detection
            results = self.model(image_path, conf=self.config["confidence"])

            # Extract detections
            detections = extract_detections(results)

            # Enhance detections with attributes based on the image
            detections = self._enhance_detections_with_attributes(detections, image_path)

            # Calculate relative positions if image dimensions provided
            if image_dimensions:
                from langvio.vision.utils import calculate_relative_positions, detect_spatial_relationships
                detections = calculate_relative_positions(detections, *image_dimensions)
                detections = detect_spatial_relationships(detections)

            # Return ALL results without filtering (use "0" as the frame key for images)
            return {"0": detections}
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return {"0": []}

    def process_video(self, video_path: str, query_params: Dict[str, Any],
                      sample_rate: int = DEFAULT_VIDEO_SAMPLE_RATE) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a video with YOLO with enhanced activity and tracking detection.
        Modified to return all detections without filtering.

        Args:
            video_path: Path to the input video
            query_params: Parameters from the query processor
            sample_rate: Process every Nth frame

        Returns:
            Dictionary with all detection results without filtering
        """
        self.logger.info(f"Processing video: {video_path} (sample rate: {sample_rate})")

        # Load model if not already loaded
        if not self.model:
            self.initialize()

        # Open video
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")

            # Get video dimensions
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_dimensions = (width, height)

            # Process frames
            frame_detections = {}
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every Nth frame
                if frame_idx % sample_rate == 0:
                    # Save frame to temp file
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
                        temp_path = temp.name

                    cv2.imwrite(temp_path, frame)

                    # Run detection
                    results = self.model(temp_path, conf=self.config["confidence"])

                    # Extract detections
                    detections = extract_detections(results)

                    # Enhance detections with attributes
                    detections = self._enhance_detections_with_attributes(detections, temp_path)

                    # Calculate relative positions and relationships
                    from langvio.vision.utils import calculate_relative_positions, detect_spatial_relationships
                    detections = calculate_relative_positions(detections, *video_dimensions)
                    detections = detect_spatial_relationships(detections)

                    # Store ALL results without filtering
                    frame_detections[str(frame_idx)] = detections

                    # Clean up
                    os.remove(temp_path)

                frame_idx += 1

            cap.release()

            # Analyze for activities and tracking across frames if needed
            # but don't filter out any detections yet
            if frame_detections and query_params.get("task_type") in ["tracking", "activity"]:
                frame_detections = self._analyze_video_for_activities(frame_detections, query_params)

            return frame_detections
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            return {}