"""
langvio: Connect language models to vision models for natural language visual analysis
"""

__version__ = "0.1.0"

from langvio.core.pipeline import Pipeline
from langvio.core.registry import ModelRegistry

# Initialize the global model registry
registry = ModelRegistry()

# Import main components for easier access
from langvio.llm.base import BaseLLMProcessor
from langvio.vision.base import BaseVisionProcessor


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

    pipeline = Pipeline()

    if config_path:
        pipeline.load_config(config_path)

    if llm_name:
        pipeline.set_llm_processor(llm_name)

    if vision_name:
        pipeline.set_vision_processor(vision_name)

    return pipeline


# Register default models
from langvio.llm.langchain_models import OpenAIProcessor
from langvio.vision.yolo.detector import YOLOProcessor

# Version info
__all__ = [
    "Pipeline",
    "create_pipeline",
    "registry",
    "BaseLLMProcessor",
    "BaseVisionProcessor"
]