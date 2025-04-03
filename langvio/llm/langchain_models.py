"""
Core LLM processor using LangChain
"""

import json
import logging
from typing import Dict, Any, List, Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatModel

from langvio.llm.base import BaseLLMProcessor


class LangChainProcessor(BaseLLMProcessor):
    """LLM processor using LangChain models"""

    def __init__(self, name: str = "langchain",
                 model_name: str = "gpt-3.5-turbo",
                 api_configs: Optional[Dict[str, Any]] = None,
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize LangChain processor.

        Args:
            name: Processor name
            model_name: Name of the model to use (e.g., "gpt-3.5-turbo", "claude-3-opus", "gemini-pro")
            api_configs: API configuration parameters (API keys, etc.)
            model_kwargs: Additional model parameters (temperature, etc.)
            **kwargs: Additional processor parameters
        """
        config = {
            "model_name": model_name,
            "api_configs": api_configs or {},
            "model_kwargs": model_kwargs or {},
            **kwargs
        }
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.llm = None
        self.query_chain = None
        self.explanation_chain = None

    def initialize(self) -> bool:
        """
        Initialize the processor with the appropriate LangChain model.

        Returns:
            True if initialization was successful
        """
        try:
            # Get model configuration
            model_name = self.config["model_name"]
            model_kwargs = self.config["model_kwargs"]
            api_configs = self.config["api_configs"]

            # Set environment variables for API keys if provided
            self._setup_api_environment(api_configs)

            # Create the LLM using LangChain's Chat Model
            self.llm = ChatModel.from_model_name(
                model_name=model_name,
                **model_kwargs
            )

            self._setup_prompts()
            return True
        except Exception as e:
            self.logger.error(f"Error initializing LangChain processor: {e}")
            return False

    def _setup_api_environment(self, api_configs: Dict[str, Any]) -> None:
        """
        Set up environment variables for API keys.
        LangChain will use these automatically.

        Args:
            api_configs: API configuration parameters
        """
        import os

        # Set environment variables for common API providers
        if "openai_api_key" in api_configs:
            os.environ["OPENAI_API_KEY"] = api_configs["openai_api_key"]

        if "anthropic_api_key" in api_configs:
            os.environ["ANTHROPIC_API_KEY"] = api_configs["anthropic_api_key"]

        if "google_api_key" in api_configs:
            os.environ["GOOGLE_API_KEY"] = api_configs["google_api_key"]

        # Set any other API environment variables
        for key, value in api_configs.items():
            if key.endswith("_api_key") and not key in ["openai_api_key", "anthropic_api_key", "google_api_key"]:
                env_key = key.upper()
                os.environ[env_key] = value

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