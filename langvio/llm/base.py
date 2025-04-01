"""
Base classes for LLM processors
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from langvio.core.base import Processor


class BaseLLMProcessor(Processor):
    """Base class for all LLM processors"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM processor.

        Args:
            name: Processor name
            config: Configuration parameters
        """
        super().__init__(name, config)
        self.model = None

    @abstractmethod
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a natural language query into structured parameters.

        Args:
            query: Natural language query

        Returns:
            Dictionary with structured parameters
        """
        pass

    @abstractmethod
    def generate_explanation(self, query: str, detections: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Generate an explanation based on detection results.

        Args:
            query: Original query
            detections: Detection results

        Returns:
            Human-readable explanation
        """
        pass

    def _extract_target_objects(self, query: str) -> List[str]:
        """
        Extract target objects from a query (fallback method).

        Args:
            query: Natural language query

        Returns:
            List of target object names
        """
        # Common object categories in COCO dataset
        common_objects = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

        # Find which common objects appear in the query
        query_lower = query.lower()
        found_objects = []

        for obj in common_objects:
            if obj in query_lower:
                found_objects.append(obj)

        # If no objects found, return a default
        if not found_objects:
            return ["person", "car"]

        return found_objects