import pytest

from langvio.core.registry import ModelRegistry


class MockProcessor:
    def __init__(self, name="mock", **kwargs):
        self.name = name
        self.kwargs = kwargs

    def initialize(self):
        return True


class MockLLMProcessor(MockProcessor):
    pass


class MockVisionProcessor(MockProcessor):
    pass


def test_registry_initialization():
    """Test registry initialization."""
    registry = ModelRegistry()
    assert registry._llm_processors == {}
    assert registry._vision_processors == {}


def test_llm_processor_registration():
    """Test registering and retrieving LLM processors."""
    registry = ModelRegistry()

    # Register an LLM processor
    registry.register_llm_processor(
        "test-llm", MockLLMProcessor, test_param="test_value"
    )

    # Check if it's registered
    processors = registry.list_llm_processors()
    assert "test-llm" in processors
    assert processors["test-llm"] == MockLLMProcessor

    # Get the processor instance
    processor = registry.get_llm_processor("test-llm")
    assert isinstance(processor, MockLLMProcessor)
    assert processor.kwargs["test_param"] == "test_value"

    # Override parameter when getting
    processor = registry.get_llm_processor("test-llm", test_param="new_value")
    assert processor.kwargs["test_param"] == "new_value"


def test_vision_processor_registration():
    """Test registering and retrieving vision processors."""
    registry = ModelRegistry()

    # Register a vision processor
    registry.register_vision_processor(
        "test-vision", MockVisionProcessor, test_param="test_value"
    )

    # Check if it's registered
    processors = registry.list_vision_processors()
    assert "test-vision" in processors
    assert processors["test-vision"] == MockVisionProcessor

    # Get the processor instance
    processor = registry.get_vision_processor("test-vision")
    assert isinstance(processor, MockVisionProcessor)
    assert processor.kwargs["test_param"] == "test_value"

    # Override parameter when getting
    processor = registry.get_vision_processor("test-vision", test_param="new_value")
    assert processor.kwargs["test_param"] == "new_value"


def test_get_nonexistent_processor():
    """Test getting a processor that doesn't exist."""
    registry = ModelRegistry()

    # Getting non-existent LLM processor should raise ValueError
    with pytest.raises(ValueError, match="LLM processor 'nonexistent' not registered"):
        registry.get_llm_processor("nonexistent")

    # Getting non-existent vision processor should raise ValueError
    with pytest.raises(
        ValueError, match="Vision processor 'nonexistent' not registered"
    ):
        registry.get_vision_processor("nonexistent")


def test_multiple_processors():
    """Test registering and retrieving multiple processors."""
    registry = ModelRegistry()

    # Register multiple LLM processors
    registry.register_llm_processor("llm1", MockLLMProcessor, param1="value1")
    registry.register_llm_processor("llm2", MockLLMProcessor, param1="value2")

    # Register multiple vision processors
    registry.register_vision_processor("vision1", MockVisionProcessor, param1="value1")
    registry.register_vision_processor("vision2", MockVisionProcessor, param1="value2")

    # Check listings
    llm_processors = registry.list_llm_processors()
    vision_processors = registry.list_vision_processors()

    assert len(llm_processors) == 2
    assert len(vision_processors) == 2
    assert "llm1" in llm_processors
    assert "llm2" in llm_processors
    assert "vision1" in vision_processors
    assert "vision2" in vision_processors
