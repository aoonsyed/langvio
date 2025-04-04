"""
Base classes for LLM processors with improved modular design
"""

import json
import logging
import importlib.util
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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
        self.query_prompt = PromptTemplate(
            input_variables=["query"],
            template=QUERY_PARSING_TEMPLATE
        )

        # Explanation prompt
        self.explanation_prompt = PromptTemplate(
            input_variables=["query", "detection_summary"],
            template=EXPLANATION_TEMPLATE
        )

        # Video analysis prompt
        self.video_analysis_prompt = PromptTemplate(
            input_variables=["query", "video_length", "frames_analyzed",
                            "frame_rate", "detection_summary", "frame_details"],
            template=VIDEO_ANALYSIS_TEMPLATE
        )



        # Create chains
        self.query_chain = LLMChain(llm=self.llm, prompt=self.query_prompt)
        self.explanation_chain = LLMChain(llm=self.llm, prompt=self.explanation_prompt)
        self.video_analysis_chain = LLMChain(llm=self.llm, prompt=self.video_analysis_prompt)

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
            response = self.query_chain.run(query=query)
            parsed = json.loads(response.strip())
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

    def analyze_video(self, query: str, detections: Dict[str, List[Dict[str, Any]]],
                      video_info: Dict[str, Any]) -> str:
        """
        Generate a temporal analysis for video detections.

        Args:
            query: Original query
            detections: Detection results
            video_info: Video metadata like duration, fps, etc.

        Returns:
            Video analysis explanation
        """
        self.logger.info("Generating video analysis")

        # Extract video metadata
        video_length = video_info.get("duration", 0)
        frame_rate = video_info.get("fps", 30)
        frames_analyzed = len(detections)

        # Create detection summary
        detection_summary = self._summarize_detections(detections)

        # Create frame-by-frame details
        frame_details = self._create_frame_details(detections)

        try:
            analysis = self.video_analysis_chain.run(
                query=query,
                video_length=video_length,
                frames_analyzed=frames_analyzed,
                frame_rate=frame_rate,
                detection_summary=detection_summary,
                frame_details=frame_details
            )
            return analysis.strip()
        except Exception as e:
            self.logger.error(f"Error generating video analysis: {e}")
            return f"Video analysis could not be generated. Analyzed {frames_analyzed} frames."

    def analyze_spatial_relationships(self, query: str, detections: Dict[str, List[Dict[str, Any]]],
                                    target_relationships: List[str] = None) -> str:
        """
        Analyze spatial relationships between detected objects.

        Args:
            query: Original query
            detections: Detection results
            target_relationships: Specific relationships to analyze

        Returns:
            Spatial analysis explanation
        """
        self.logger.info("Analyzing spatial relationships between objects")

        # Format object positions
        object_positions = self._format_object_positions(detections)

        # Default target relationships if none specified
        if not target_relationships:
            target_relationships = ["next to", "above", "below", "inside", "contains"]

        try:
            analysis = self.spatial_analysis_chain.run(
                query=query,
                object_positions=object_positions,
                target_relationships=", ".join(target_relationships)
            )
            return analysis.strip()
        except Exception as e:
            self.logger.error(f"Error generating spatial analysis: {e}")
            return "Could not analyze spatial relationships between objects."

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

    def _create_frame_details(self, detections: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Create a frame-by-frame summary of notable detections.

        Args:
            detections: Detection results

        Returns:
            Frame details string
        """
        frame_details = []

        # Get a subset of frames to avoid overwhelming the LLM
        frame_keys = sorted(detections.keys(), key=lambda x: int(x))
        sample_keys = frame_keys[::max(1, len(frame_keys) // 10)]  # Sample ~10 frames

        for frame_id in sample_keys:
            frame_detections = detections[frame_id]
            if not frame_detections:
                continue

            # Format frame summary
            frame_lines = [f"Frame {frame_id}:"]
            for det in frame_detections:
                conf_pct = f"{det['confidence']*100:.1f}%"
                frame_lines.append(f"- {det['label']} ({conf_pct}) at position {det['bbox']}")

            frame_details.append("\n".join(frame_lines))

        return "\n\n".join(frame_details)

    def _format_object_positions(self, detections: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Format object positions for spatial analysis.

        Args:
            detections: Detection results

        Returns:
            Formatted object positions string
        """
        positions = []

        # For images, use the first frame
        frame_id = "0" if "0" in detections else list(detections.keys())[0]
        frame_detections = detections[frame_id]

        for i, det in enumerate(frame_detections):
            x1, y1, x2, y2 = det["bbox"]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1

            positions.append(
                f"Object {i+1}: {det['label']} at center ({center_x}, {center_y}), "
                f"width {width}, height {height}"
            )

        return "\n".join(positions)

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