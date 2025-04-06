"""
Base classes for LLM processors with streamlined design
"""

import json
import logging
import importlib.util
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.json import SimpleJsonOutputParser

from langvio.core.base import Processor
from langvio.prompts import (
    QUERY_PARSING_TEMPLATE,
    EXPLANATION_TEMPLATE,
    SYSTEM_PROMPT
)

class BaseLLMProcessor(Processor):
    """Base class for all LLM processors"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM processor."""
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.llm = None
        self.query_chat_prompt = None
        self.explanation_chat_prompt = None

    def initialize(self) -> bool:
        """Initialize the processor with its configuration."""
        try:
            # Set up API environment variables if provided
            if "api_configs" in self.config:
                self._setup_api_environment(self.config.get("api_configs", {}))

            # Initialize the specific LLM implementation
            self._initialize_llm()

            # Set up prompts
            self._setup_prompts()

            return True
        except Exception as e:
            self.logger.error(f"Error initializing LLM processor: {e}")
            return False

    @abstractmethod
    def _initialize_llm(self) -> None:
        """Initialize the specific LLM implementation."""
        pass

    def _setup_api_environment(self, api_configs: Dict[str, Any]) -> None:
        """Set up environment variables for API keys."""
        import os
        for key, value in api_configs.items():
            if key.endswith("_api_key") and value:
                os.environ[key.upper()] = value

    def _setup_prompts(self) -> None:
        """Set up the prompt templates with system message."""
        system_message = SystemMessage(content=SYSTEM_PROMPT)

        # Query parsing prompt
        self.query_chat_prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="history"),
            # ("user",f"TASK: PARSE QUERY\n\n{QUERY_PARSING_TEMPLATE} \n\n{FORMAT_INSTRUCTION_QUERY}")
            ("user", QUERY_PARSING_TEMPLATE)
        ])

        # Explanation prompt
        self.explanation_chat_prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="history"),
            ("user", EXPLANATION_TEMPLATE)
        ])

        # Create chains
        json_parser = SimpleJsonOutputParser()
        self.query_chain = self.query_chat_prompt | self.llm | json_parser
        self.explanation_chain = self.explanation_chat_prompt | self.llm | json_parser

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse a natural language query into structured parameters."""
        self.logger.info(f"Parsing query: {query}")

        try:


            # Invoke the chain with proper output parsing
            parsed = self.query_chain.invoke({"query": query, "history": []})

            # Ensure required fields exist
            if "target_objects" not in parsed:
                parsed["target_objects"] = []
            if "count_objects" not in parsed:
                parsed["count_objects"] = False
            if "task_type" not in parsed:
                parsed["task_type"] = "identification"
            if "attributes" not in parsed:
                parsed["attributes"] = []
            if "spatial_relations" not in parsed:
                parsed["spatial_relations"] = []

            return parsed

        except Exception as e:
            self.logger.error(f"Error parsing query: {e}")

            # Fallback to simple extraction
            target_objects = self._extract_target_objects(query)
            counting = any(word in query.lower() for word in ["count", "how many", "number of"])

            return {
                "target_objects": target_objects,
                "count_objects": counting,
                "task_type": "counting" if counting else "identification",
                "attributes": [],
                "spatial_relations": []
            }

    def generate_explanation(self, query: str, detections: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate an explanation based on detection results."""
        self.logger.info("Generating explanation for detection results")

        # Create a summary of detections
        detection_summary = self._summarize_detections(detections)

        try:
            # Invoke the model with the explanation chat prompt
            # Invoke the explanation chain
            response = self.explanation_chain.invoke({
                "query": query,
                "detection_summary": detection_summary,
                "history": []
            })

            return response

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

            explanation = "Analysis results: " + ", ".join(
                f"{count} {label}{'s' if count > 1 else ''}"
                for label, count in object_counts.items()
            )
            return explanation + "."

    def _summarize_detections(self, detections: Dict[str, List[Dict[str, Any]]]) -> str:
        """Create a summary of detections for the LLM to explain."""
        # Count objects by label
        label_counts = {}
        total_frames = len(detections)

        for frame_detections in detections.values():
            for det in frame_detections:
                label = det["label"]
                label_counts[label] = label_counts.get(label, 0) + 1

        # Format summary
        summary_lines = [
            f"{label}: {count} instances detected"
            for label, count in label_counts.items()
        ]

        # Add media type info for videos
        if total_frames > 1:
            summary_lines.append(f"Total frames analyzed: {total_frames}")

        return "\n".join(summary_lines)

    def is_package_installed(self, package_name: str) -> bool:
        """Check if a Python package is installed."""
        return importlib.util.find_spec(package_name) is not None

    def _extract_target_objects(self, query: str) -> List[str]:
        """Extract target objects from a query (fallback method)."""
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

        # Find objects in the query
        query_lower = query.lower()
        found_objects = [obj for obj in common_objects if obj in query_lower]

        # Return defaults if no objects found
        return found_objects if found_objects else ["person", "car"]