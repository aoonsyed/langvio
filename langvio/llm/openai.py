"""
OpenAI-specific LLM processor implementation
"""

import logging
from typing import Dict, Any, Optional

from langvio.llm.base import BaseLLMProcessor


class OpenAIProcessor(BaseLLMProcessor):
    """LLM processor using OpenAI models via LangChain"""

    def __init__(self, name: str = "openai",
                 model_name: str = "gpt-3.5-turbo",
                 api_configs: Optional[Dict[str, Any]] = None,
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize OpenAI processor.

        Args:
            name: Processor name
            model_name: Name of the OpenAI model to use
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

    def _initialize_llm(self) -> None:
        """
        Initialize the OpenAI model via LangChain.
        This is the only method that needs to be implemented.
        """
        try:
            # Check if OpenAI and LangChain OpenAI wrapper are installed
            if not self.is_package_installed("openai"):
                raise ImportError(
                    "The 'openai' package is required to use OpenAI models. "
                    "Please install it with 'pip install langvio[openai]'"
                )

            if not self.is_package_installed("langchain_openai"):
                raise ImportError(
                    "The 'langchain-openai' package is required to use OpenAI models. "
                    "Please install it with 'pip install langvio[openai]'"
                )

            # Import necessary components
            from langchain_openai import ChatOpenAI

            # Get model configuration
            model_name = self.config["model_name"]
            model_kwargs = self.config["model_kwargs"].copy()

            # Ensure API key is in environment
            if "api_configs" in self.config and "openai_api_key" in self.config["api_configs"]:
                import os
                os.environ["OPENAI_API_KEY"] = self.config["api_configs"]["openai_api_key"]

            # Create the OpenAI LLM
            self.llm = ChatOpenAI(
                model_name=model_name,
                **model_kwargs
            )

            self.logger.info(f"Initialized OpenAI model: {model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing OpenAI model: {e}")
            raise