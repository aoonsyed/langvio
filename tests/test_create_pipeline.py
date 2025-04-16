import pytest
from unittest.mock import patch, MagicMock
import tempfile
import yaml
import os
from langvio import create_pipeline
from langvio.core.pipeline import Pipeline


# Mock classes for testing
class MockLLMProcessor:
    def __init__(self, name="mock-llm", **kwargs):
        self.name = name
        self.kwargs = kwargs

    def initialize(self):
        return True


class MockVisionProcessor:
    def __init__(self, name="mock-vision", **kwargs):
        self.name = name
        self.kwargs = kwargs

    def initialize(self):
        return True


@pytest.fixture
def mock_registry():
    """Create a mocked registry."""
    mock = MagicMock()
    mock.list_llm_processors.return_value = {"default": MockLLMProcessor, "test-llm": MockLLMProcessor}
    mock.list_vision_processors.return_value = {"yolo": MockVisionProcessor}
    mock.get_llm_processor.return_value = MockLLMProcessor()
    mock.get_vision_processor.return_value = MockVisionProcessor()
    return mock


@pytest.fixture
def test_config():
    """Create a test configuration file."""
    config = {
        "llm": {
            "default": "test-llm",
            "models": {
                "test-llm": {
                    "model_name": "test-model",
                    "model_kwargs": {"temperature": 0.5}
                }
            }
        },
        "vision": {
            "default": "test-vision",
            "models": {
                "test-vision": {
                    "type": "test",
                    "model_path": "test-model.pt",
                    "confidence": 0.3
                }
            }
        }
    }

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp:
        temp_path = temp.name
        yaml.dump(config, temp)

    return temp_path


@patch('langvio.registry')
def test_create_pipeline_defaults(mock_registry):
    """Test creating a pipeline with default settings."""
    # Set up mock registry
    mock_registry.list_llm_processors.return_value = {"default": MockLLMProcessor}
    mock_registry.list_vision_processors.return_value = {"yolo": MockVisionProcessor}
    mock_registry.get_llm_processor.return_value = MockLLMProcessor()
    mock_registry.get_vision_processor.return_value = MockVisionProcessor()

    # Create pipeline
    pipeline = create_pipeline()

    # Check that it's a Pipeline instance
    assert isinstance(pipeline, Pipeline)

    # Check that the processors were set
    assert pipeline.llm_processor is not None
    assert pipeline.vision_processor is not None

    # Check that the vision processor is YOLO by default
    mock_registry.get_vision_processor.assert_called_with("yolo")


@patch('langvio.registry')
def test_create_pipeline_with_config(mock_registry, test_config):
    """Test creating a pipeline with a custom configuration."""
    # Set up mock registry
    mock_registry.list_llm_processors.return_value = {"test-llm": MockLLMProcessor}
    mock_registry.list_vision_processors.return_value = {"test-vision": MockVisionProcessor}
    mock_registry.get_llm_processor.return_value = MockLLMProcessor()
    mock_registry.get_vision_processor.return_value = MockVisionProcessor()

    try:
        # Create pipeline with config
        pipeline = create_pipeline(config_path=test_config)

        # Check that it's a Pipeline instance
        assert isinstance(pipeline, Pipeline)

        # Check that it loaded the configuration
        assert pipeline.config.config["llm"]["default"] == "test-llm"
        assert pipeline.config.config["vision"]["default"] == "test-vision"
    finally:
        # Clean up
        os.unlink(test_config)


@patch('langvio.registry')
def test_create_pipeline_with_specified_processors(mock_registry):
    """Test creating a pipeline with specified processors."""
    # Set up mock registry
    mock_registry.list_llm_processors.return_value = {
        "default": MockLLMProcessor,
        "gpt-4": MockLLMProcessor,
        "gemini": MockLLMProcessor
    }
    mock_registry.list_vision_processors.return_value = {
        "yolo": MockVisionProcessor,
        "yolo_large": MockVisionProcessor
    }
    mock_registry.get_llm_processor.return_value = MockLLMProcessor()
    mock_registry.get_vision_processor.return_value = MockVisionProcessor()

    # Create pipeline with specified processors
    pipeline = create_pipeline(llm_name="gpt-4", vision_name="yolo_large")

    # Check that it's a Pipeline instance
    assert isinstance(pipeline, Pipeline)

    # Check that the correct processors were requested
    mock_registry.get_llm_processor.assert_called_with("gpt-4")
    mock_registry.get_vision_processor.assert_called_with("yolo_large")


@patch('langvio.registry')
def test_create_pipeline_with_no_llm_providers(mock_registry):
    """Test error handling when no LLM providers are available."""
    # Set up mock registry to simulate no LLM providers
    mock_registry.list_llm_processors.return_value = {}
    mock_registry.list_vision_processors.return_value = {"yolo": MockVisionProcessor}

    # Should exit with an error
    with pytest.raises(SystemExit):
        create_pipeline()


@patch('langvio.registry')
def test_create_pipeline_with_nonexistent_provider(mock_registry):
    """Test error handling with a non-existent provider."""
    # Set up mock registry
    mock_registry.list_llm_processors.return_value = {"default": MockLLMProcessor}
    mock_registry.list_vision_processors.return_value = {"yolo": MockVisionProcessor}
    mock_registry.get_llm_processor.return_value = MockLLMProcessor()
    mock_registry.get_vision_processor.return_value = MockVisionProcessor()

    # Mock get_llm_processor to raise an exception for non-existent model
    mock_registry.get_llm_processor.side_effect = lambda name, **kwargs: (
        MockLLMProcessor() if name != "nonexistent" else SystemExit(1)
    )

    # Should exit with an error for non-existent LLM
    with pytest.raises(SystemExit):
        create_pipeline(llm_name="nonexistent")