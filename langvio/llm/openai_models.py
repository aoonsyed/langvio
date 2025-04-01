"""
OpenAI-based LLM processors
"""

import logging
from typing import Dict, Any, List, Optional

from langchain.chat_models import ChatOpenAI

from langvio.llm.langchain_models import LangChainProcessor


class OpenAIProcessor(LangChainProcessor):
    """LLM processor using OpenAI models via LangChain"""

    def __init__(self, name: str = "openai", model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.0, **kwargs):
        """
        Initialize OpenAI processor.

        Args:
            name: Processor name
            model_name: Name of the OpenAI model to use
            temperature: Temperature for sampling
            **kwargs: Additional parameters for the OpenAI API
        """
        super().__init__(name, model_name, temperature, **kwargs)
        self.logger = logging.getLogger(__name__)

    def _setup_llm(self) -> bool:
        """
        Set up the OpenAI LLM and prompt chains.

        Returns:
            True if setup was successful
        """
        try:
            self.llm = ChatOpenAI(
                model_name=self.config["model_name"],
                temperature=self.config["temperature"]
            )

            self._setup_prompts()
            return True
        except Exception as e:
            self.logger.error(f"Error setting up OpenAI LLM: {e}")
            return False


# Register with the global registry
from langvio import registry

registry.register_llm_processor("langchain_openai", OpenAIProcessor)