"""
Base classes for vision processors
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from langvio.core.base import Processor


class BaseVisionProcessor(Processor):
    """Base class for all vision processors"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize vision processor.

        Args:
            name: Processor name
            config: Configuration parameters
        """
        super().__init__(name, config)
        self.model = None

    @abstractmethod
    def process_image(self, image_path: str, query_params: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process an image with the vision model.

        Args:
            image_path: Path to the input image
            query_params: Parameters from the query processor

        Returns:
            Dictionary with detection results
        """
        pass

    @abstractmethod
    def process_video(self, video_path: str, query_params: Dict[str, Any],
                      sample_rate: int = 1) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a video with the vision model.

        Args:
            video_path: Path to the input video
            query_params: Parameters from the query processor
            sample_rate: Process every Nth frame

        Returns:
            Dictionary with detection results
        """
        pass

    def _filter_detections(self, detections: List[Dict[str, Any]],
                           query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter detections based on query parameters.

        Args:
            detections: Raw detection results
            query_params: Query parameters

        Returns:
            Filtered detection results
        """
        # Extract target objects
        target_objects = [obj.lower() for obj in query_params.get("target_objects", [])]

        # If no target objects specified, return all detections
        if not target_objects:
            return detections

        # Filter by target objects
        filtered = []
        for det in detections:
            if det["label"].lower() in target_objects:
                filtered.append(det)

        return filtered