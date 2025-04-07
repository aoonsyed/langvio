"""
Enhanced base classes for LLM processors with expanded capabilities
"""

import json
import logging
import importlib.util
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.json import SimpleJsonOutputParser

from langvio.core.base import Processor
from langvio.prompts.templates import (
    QUERY_PARSING_TEMPLATE,
    EXPLANATION_TEMPLATE,
    SYSTEM_PROMPT
)
from langvio.prompts.constants import (
    TASK_TYPES,
    VISUAL_ATTRIBUTES,
    SPATIAL_RELATIONS,
    ACTIVITIES,
    COMMON_OBJECTS
)

class BaseLLMProcessor(Processor):
    """Enhanced base class for all LLM processors"""

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

            # Ensure all required fields exist with defaults
            parsed = self._ensure_parsed_fields(parsed)

            # Log the parsed query
            self.logger.debug(f"Parsed query: {json.dumps(parsed, indent=2)}")

            return parsed

        except Exception as e:
            self.logger.error(f"Error parsing query: {e}")

            # Fallback to simple extraction
            return self._fallback_query_parsing(query)

    def _ensure_parsed_fields(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required fields exist in the parsed query."""
        defaults = {
            "target_objects": [],
            "count_objects": False,
            "task_type": "identification",
            "attributes": [],
            "spatial_relations": [],
            "activities": [],
            "custom_instructions": ""
        }

        # Add any missing fields with defaults
        for key, default_value in defaults.items():
            if key not in parsed or parsed[key] is None:
                parsed[key] = default_value

        # Ensure task_type is valid
        if parsed["task_type"] not in TASK_TYPES:
            self.logger.warning(f"Invalid task type: {parsed['task_type']}. Using 'identification' instead.")
            parsed["task_type"] = "identification"

        return parsed

    def _fallback_query_parsing(self, query: str) -> Dict[str, Any]:
        """Simple fallback method for query parsing when LLM fails."""
        self.logger.info("Using fallback query parsing")

        # Default values
        parsed = {
            "target_objects": [],
            "count_objects": False,
            "task_type": "identification",
            "attributes": [],
            "spatial_relations": [],
            "activities": [],
            "custom_instructions": ""
        }

        # Lowercased query for matching
        query_lower = query.lower()

        # Extract target objects
        parsed["target_objects"] = self._extract_target_objects(query)

        # Detect if counting is needed
        counting_terms = ["count", "how many", "number of", "total"]
        parsed["count_objects"] = any(term in query_lower for term in counting_terms)

        # Determine task type
        if parsed["count_objects"]:
            parsed["task_type"] = "counting"
        elif any(term in query_lower for term in ["is there", "are there", "can you see", "do you see"]):
            parsed["task_type"] = "verification"
        elif any(term in query_lower for term in ["track", "follow", "movement"]):
            parsed["task_type"] = "tracking"
        elif any(term in query_lower for term in ["activity", "doing", "action", "behavior"]):
            parsed["task_type"] = "activity"

        # Check for attribute terms
        for attr in VISUAL_ATTRIBUTES:
            if attr in query_lower:
                # Simple extraction of potential values
                words = query_lower.split()
                attr_idx = -1

                for i, word in enumerate(words):
                    if attr in word:
                        attr_idx = i
                        break

                if attr_idx >= 0 and attr_idx < len(words) - 1:
                    # Take the next word as a potential value
                    value = words[attr_idx + 1].strip(",.:;?!")
                    parsed["attributes"].append({"attribute": attr, "value": value})

        # Check for spatial relation terms
        for relation in SPATIAL_RELATIONS:
            relation_term = relation.replace("_", " ")
            if relation_term in query_lower:
                # Find potential object after the relation
                rel_idx = query_lower.find(relation_term) + len(relation_term)
                if rel_idx < len(query_lower):
                    # Extract text after the relation term
                    after_text = query_lower[rel_idx:].strip()
                    # Find the first potential object word
                    for obj in COMMON_OBJECTS:
                        if obj in after_text:
                            parsed["spatial_relations"].append({
                                "relation": relation,
                                "object": obj
                            })
                            break

        # Check for activity terms
        for activity in ACTIVITIES:
            if activity in query_lower:
                parsed["activities"].append(activity)

        return parsed

    def generate_explanation(self, query: str, detections: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate an explanation based on detection results."""
        self.logger.info("Generating explanation for detection results")

        # Get the original parsed query
        parsed_query = self.parse_query(query)

        # Create a summary of detections
        detection_summary = self._summarize_detections(detections, parsed_query)

        try:
            # Invoke the explanation chain
            response = self.explanation_chain.invoke({
                "query": query,
                "detection_summary": detection_summary,
                "parsed_query": json.dumps(parsed_query, indent=2),
                "history": []
            })

            return response

        except Exception as e:
            self.logger.error(f"Error generating explanation: {e}")
            return self._fallback_explanation(detections, parsed_query)

    def _fallback_explanation(self, detections: Dict[str, List[Dict[str, Any]]],
                             parsed_query: Dict[str, Any]) -> str:
        """Simple fallback method for explanation generation when LLM fails."""
        self.logger.info("Using fallback explanation generation")

        # Get counts of detected objects
        object_counts = {}
        for frame_detections in detections.values():
            for det in frame_detections:
                label = det["label"]
                object_counts[label] = object_counts.get(label, 0) + 1

        # Create a simple explanation based on task type
        task_type = parsed_query.get("task_type", "identification")
        target_objects = parsed_query.get("target_objects", [])

        if not object_counts:
            # No detections
            if target_objects:
                objects_str = ", ".join(target_objects)
                return f"I did not detect any {objects_str} in the {'video' if len(detections) > 1 else 'image'}."
            else:
                return "No objects of interest were detected."

        if task_type == "counting":
            # For counting tasks
            count_explanations = []
            for label, count in object_counts.items():
                if not target_objects or label in target_objects:
                    count_explanations.append(f"{count} {label}{'s' if count > 1 else ''}")

            return "I found " + ", ".join(count_explanations) + "."

        elif task_type == "verification":
            # For verification tasks
            verified_objects = [label for label in target_objects if label in object_counts]

            if verified_objects:
                return f"Yes, I found {', '.join(verified_objects)} in the {'video' if len(detections) > 1 else 'image'}."
            else:
                return f"No, I did not find {', '.join(target_objects)} in the {'video' if len(detections) > 1 else 'image'}."

        elif task_type == "tracking" or task_type == "activity":
            # For tracking/activity tasks
            activity_str = ""
            if parsed_query.get("activities"):
                activity_str = f" {', '.join(parsed_query['activities'])}"

            object_str = ", ".join(target_objects) if target_objects else "objects"

            return f"I tracked {sum(object_counts.values())} instances of {object_str}{activity_str} across {len(detections)} frames."

        else:
            # Default identification task
            explanation = "Analysis results: " + ", ".join(
                f"{count} {label}{'s' if count > 1 else ''}"
                for label, count in object_counts.items()
                if not target_objects or label in target_objects
            )

            return explanation + "."