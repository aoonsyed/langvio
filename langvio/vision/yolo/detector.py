"""
YOLO-based vision processor
"""

import os
import tempfile
import logging
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from langvio.vision.base import BaseVisionProcessor
from langvio.vision.yolo.utils import extract_detections, filter_by_confidence


class YOLOProcessor(BaseVisionProcessor):
    """Vision processor using YOLO models"""

    def __init__(self, name: str = "yolo", model_path: str = "yolov8n.pt",
                 confidence: float = 0.25, **kwargs):
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
        Process an image with YOLO.

        Args:
            image_path: Path to the input image
            query_params: Parameters from the query processor

        Returns:
            Dictionary with detection results
        """
        self.logger.info(f"Processing image: {image_path}")

        # Load model if not already loaded
        if not self.model:
            self.initialize()

        # Run detection
        try:
            results = self.model(image_path, conf=self.config["confidence"])

            # Extract and filter detections
            detections = extract_detections(results)
            filtered_detections = self._filter_detections(detections, query_params)

            # Return results (use "0" as the frame key for images)
            return {"0": filtered_detections}
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return {"0": []}

    def process_video(self, video_path: str, query_params: Dict[str, Any],
                      sample_rate: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a video with YOLO.

        Args:
            video_path: Path to the input video
            query_params: Parameters from the query processor
            sample_rate: Process every Nth frame

        Returns:
            Dictionary with detection results
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

                    # Extract and filter detections
                    detections = extract_detections(results)
                    filtered_detections = self._filter_detections(detections, query_params)

                    # Store results
                    frame_detections[str(frame_idx)] = filtered_detections

                    # Clean up
                    os.remove(temp_path)

                frame_idx += 1

            cap.release()

            return frame_detections
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            return {}


# Register with the global registry
from langvio import registry

registry.register_vision_processor("yolo", YOLOProcessor)