"""
Base classes for LLM processors with improved modular design
"""

import json
import logging
import importlib.util
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

from langvio.core.base import Processor
from langvio.prompts import (
    QUERY_PARSING_TEMPLATE,
    EXPLANATION_TEMPLATE,
    VIDEO_ANALYSIS_TEMPLATE
)


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
        self.logger = logging.getLogger(__name__)
        self.llm = None
        self.query_chain = None
        self.explanation_chain = None
        self.video_analysis_chain = None
        self.spatial_analysis_chain = None

    def initialize(self) -> bool:
        """
        Initialize the processor with its configuration.
        This is a concrete implementation that calls the abstract _initialize_llm() method.

        Returns:
            True if initialization was successful
        """
        try:
            # Set up API environment variables
            if "api_configs" in self.config:
                self._setup_api_environment(self.config.get("api_configs", {}))

            # Initialize the specific LLM implementation (implemented by subclasses)
            self._initialize_llm()

            # Set up prompts and chains
            self._setup_prompts()

            return True
        except Exception as e:
            self.logger.error(f"Error initializing LLM processor: {e}")
            return False

    @abstractmethod
    def _initialize_llm(self) -> None:
        """
        Initialize the specific LLM implementation.
        This is the only method that subclasses must implement.
        """
        pass

    def _setup_api_environment(self, api_configs: Dict[str, Any]) -> None:
        """
        Set up environment variables for API keys.

        Args:
            api_configs: API configuration parameters
        """
        import os

        # Set environment variables for API keys
        for key, value in api_configs.items():
            if key.endswith("_api_key") and value:
                env_key = key.upper()
                os.environ[env_key] = value

    def _setup_prompts(self) -> None:
        """Set up the prompt templates and chains"""
        # Query parsing prompt
        self.query_prompt = ChatPromptTemplate.from_template(
            template=QUERY_PARSING_TEMPLATE
        )

        self.explanation_prompt = ChatPromptTemplate.from_template(
            template=EXPLANATION_TEMPLATE
        )



        # Create chains
        self.query_chain = self.query_prompt | self.llm  # | output_parser # Optional parser
        self.explanation_chain = self.explanation_prompt | self.llm  # | output_parser # Optional parser

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a natural language query into structured parameters.

        Args:
            query: Natural language query

        Returns:
            Dictionary with structured parameters
        """
        self.logger.info(f"Parsing query: {query}")

        try:
            input_data = {"query": query}
            response = self.query_chain.invoke(input_data)
            parsed = json.loads(response.content.strip())
            return parsed
        except Exception as e:
            self.logger.error(f"Error parsing query: {e}")

            # Fallback: extract target objects from query directly
            target_objects = self._extract_target_objects(query)

            # Determine if query is asking for counting
            counting = any(word in query.lower() for word in
                           ["count", "how many", "number of"])

            return {
                "target_objects": target_objects,
                "count_objects": counting,
                "task_type": "counting" if counting else "identification",
                "attributes": [],
                "spatial_relations": []
            }

    def generate_explanation(self, query: str, detections: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Generate an explanation based on detection results.

        Args:
            query: Original query
            detections: Detection results

        Returns:
            Human-readable explanation
        """
        self.logger.info("Generating explanation for detection results")

        # Create a summary of detections
        detection_summary = self._summarize_detections(detections)

        try:
            explanation = self.explanation_chain.run(
                query=query,
                detection_summary=detection_summary
            )
            return explanation.strip()
        except Exception as e:
            self.logger.error(f"Error generating explanation: {e}")

            # Simple fallback explanation
            object_counts = {}
            for frame_detections in detections.values():
                for det in frame_detections:
                    label = det["label"]
                    object_counts[label] = object_counts.get(label, 0) + 1

            if not object_counts:
                return "No objects of interest were detected."

            # Format basic explanation
            explanation = "Analysis results: "
            for label, count in object_counts.items():
                explanation += f"{count} {label}{'s' if count > 1 else ''}, "

            return explanation[:-2] + "."




    def _summarize_detections(self, detections: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Create a summary of detections for the LLM to explain.

        Args:
            detections: Detection results

        Returns:
            Summary string
        """
        # Count objects by label
        label_counts = {}
        total_frames = len(detections)

        for frame_id, frame_detections in detections.items():
            for det in frame_detections:
                label = det["label"]
                label_counts[label] = label_counts.get(label, 0) + 1

        # Format summary
        summary_lines = []

        # Add total counts
        for label, count in label_counts.items():
            summary_lines.append(f"{label}: {count} instances detected")

        # Add media type info
        if total_frames > 1:
            summary_lines.append(f"Total frames analyzed: {total_frames}")

        return "\n".join(summary_lines)

    def is_package_installed(self, package_name: str) -> bool:
        """
        Check if a Python package is installed.

        Args:
            package_name: Name of the package to check

        Returns:
            True if the package is installed, False otherwise
        """
        return importlib.util.find_spec(package_name) is not None

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