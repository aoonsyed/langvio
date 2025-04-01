"""
LLM processors using LangChain
"""

import json
import logging
from typing import Dict, Any, List, Optional

from langchain.llms import BaseLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langvio.llm.base import BaseLLMProcessor


class LangChainProcessor(BaseLLMProcessor):
    """LLM processor using LangChain"""

    def __init__(self, name: str = "langchain", model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.0, **kwargs):
        """
        Initialize LangChain processor.

        Args:
            name: Processor name
            model_name: Name of the language model to use
            temperature: Temperature for LLM sampling
            **kwargs: Additional parameters for the LLM
        """
        config = {
            "model_name": model_name,
            "temperature": temperature,
            **kwargs
        }
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.llm = None
        self.query_chain = None
        self.explanation_chain = None

    def initialize(self) -> bool:
        """
        Initialize the processor with LangChain.

        Returns:
            True if initialization was successful
        """
        try:
            # Abstract method - should be implemented by subclasses
            return self._setup_llm()
        except Exception as e:
            self.logger.error(f"Error initializing LangChain processor: {e}")
            return False

    def _setup_llm(self) -> bool:
        """
        Set up the LLM and prompt chains.

        Returns:
            True if setup was successful
        """
        # This should be implemented by subclasses
        raise NotImplementedError

    def _setup_prompts(self) -> None:
        """Set up the prompt templates and chains"""
        # Query parsing prompt
        self.query_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Translate the following natural language query about images/videos into structured commands for an object detection system.

            Query: {query}

            Return a JSON with these fields:
            - target_objects: List of object categories to detect (e.g., "person", "car", "dog", etc.)
            - count_objects: Boolean indicating if counting is needed
            - task_type: One of "identification", "counting", "verification", "analysis"
            - attributes: Any specific attributes to look for (e.g., "color", "size", "activity")

            JSON response:
            """
        )

        # Explanation prompt
        self.explanation_prompt = PromptTemplate(
            input_variables=["query", "detection_summary"],
            template="""
            Based on the user's query and detection results, provide a concise explanation.

            User query: {query}

            Detection results: {detection_summary}

            Provide a clear, helpful explanation that directly addresses the user's query based on what was detected.
            Focus on answering their specific question or fulfilling their request.

            Explanation:
            """
        )

        # Create chains
        self.query_chain = LLMChain(llm=self.llm, prompt=self.query_prompt)
        self.explanation_chain = LLMChain(llm=self.llm, prompt=self.explanation_prompt)

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
                "attributes": []
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