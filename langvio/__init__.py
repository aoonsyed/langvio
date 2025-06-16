"""
langvio: Connect language models to vision models for natural language visual analysis
"""

# Top-level imports
import sys

from langvio.core.registry import ModelRegistry
from langvio.core.pipeline import Pipeline
from langvio.llm.base import BaseLLMProcessor
from langvio.vision.base import BaseVisionProcessor
from langvio.vision.yolo.detector import YOLOProcessor
from langvio.llm.factory import register_llm_processors

__version__ = "0.3.0"

# Initialize the global model registry
registry = ModelRegistry()

# Register vision processors
registry.register_vision_processor("yolo", YOLOProcessor)
registry.register_vision_processor("yoloe_large", YOLOProcessor)
registry.register_vision_processor("yoloe", YOLOProcessor)

# Register LLM processors using the factory
register_llm_processors(registry)


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

    # Create the pipeline
    pipeline = Pipeline(config_path)

    # Set the vision processor (YOLO is always available)
    if vision_name:
        pipeline.set_vision_processor(vision_name)
    else:
        pipeline.set_vision_processor("yoloe_large")

    # Set the LLM processor if specified
    if llm_name:
        pipeline.set_llm_processor(llm_name)
    else:
        try:
            default_llm = pipeline.config.config["llm"]["default"]
            pipeline.set_llm_processor(default_llm)
        except Exception:
            if len(registry.list_llm_processors()) == 0:
                error_msg = (
                    "ERROR: No LLM providers are installed. Please install at least one provider:\n"
                    "- For OpenAI: pip install langvio[openai]\n"
                    "- For Google Gemini: pip install langvio[google]\n"
                    "- For all providers: pip install langvio[all-llm]"
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)
            else:
                available_llm = next(iter(registry.list_llm_processors()))
                pipeline.set_llm_processor(available_llm)

    return pipeline


# Public API
__all__ = [
    "Pipeline",
    "create_pipeline",
    "registry",
    "BaseLLMProcessor",
    "BaseVisionProcessor",
]
