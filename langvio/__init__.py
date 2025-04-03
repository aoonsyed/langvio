"""
langvio: Connect language models to vision models for natural language visual analysis
"""

__version__ = "0.3.0"

# Try to load environment variables from .env file
from langvio.utils.env_loader import load_dotenv
load_dotenv()

from langvio.core.pipeline import Pipeline
from langvio.core.registry import ModelRegistry

# Initialize the global model registry
registry = ModelRegistry()

# Import main components for easier access
from langvio.llm.base import BaseLLMProcessor
from langvio.vision.base import BaseVisionProcessor
from langvio.llm.langchain_models import LangChainProcessor


# Default pipeline creator
def create_pipeline(config_path=None, llm_name=None, vision_name=None):
    """
    Create a pipeline with optional configuration.

    Args:
        config_path: Path to a configuration file
        llm_name: Name of LLM processor to use
        vision_name: Name of vision processor to use

    Returns:
        A configured Pipeline instance
    """
    from langvio.core.pipeline import Pipeline

    pipeline = Pipeline(config_path)

    if llm_name:
        pipeline.set_llm_processor(llm_name)

    if vision_name:
        pipeline.set_vision_processor(vision_name)

    return pipeline


# Register default processors
from langvio.vision.yolo.detector import YOLOProcessor

# Register the LangChain processor for different model configurations
registry.register_llm_processor("gemini", LangChainProcessor, model_name="gemini-pro")  # Default to Gemini
registry.register_llm_processor("gpt", LangChainProcessor, model_name="gpt-3.5-turbo")
registry.register_llm_processor("claude", LangChainProcessor, model_name="claude-3-opus-20240229")
registry.register_llm_processor("mistral", LangChainProcessor, model_name="mistral/mistral-7b-instruct-v0.1")

# Register the YOLO processor
registry.register_vision_processor("yolo", YOLOProcessor)

# Version info
__all__ = [
    "Pipeline",
    "create_pipeline",
    "registry",
    "BaseLLMProcessor",
    "BaseVisionProcessor",
    "LangChainProcessor"
]