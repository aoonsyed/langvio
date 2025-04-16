import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch
from langvio.core.pipeline import Pipeline
from langvio.llm.base import BaseLLMProcessor
from langvio.vision.base import BaseVisionProcessor
from langvio.media.processor import MediaProcessor


class MockLLMProcessor(BaseLLMProcessor):
    """Mock LLM processor for testing."""

    def __init__(self, name="mock-llm", config=None):
        super().__init__(name, config or {})
        self._highlighted_objects = []

    def _initialize_llm(self):
        self.llm = MagicMock()
        return True

    def parse_query(self, query):
        return {
            "target_objects": ["person", "car"],
            "count_objects": False,
            "task_type": "identification",
            "attributes": [],
            "spatial_relations": [],
            "activities": [],
            "custom_instructions": ""
        }

    def generate_explanation(self, query, detections):
        self._highlighted_objects = [
            {
                "frame_key": "0",
                "detection": {"label": "person", "confidence": 0.9, "bbox": [10, 10, 50, 50]}
            }
        ]
        return "I detected a person in the image."

    def get_highlighted_objects(self):
        return self._highlighted_objects


class MockVisionProcessor(BaseVisionProcessor):
    """Mock vision processor for testing."""

    def __init__(self, name="mock-vision", config=None):
        super().__init__(name, config or {})

    def initialize(self):
        return True

    def process_image(self, image_path, query_params):
        return {
            "0": [
                {"label": "person", "confidence": 0.9, "bbox": [10, 10, 50, 50]},
                {"label": "car", "confidence": 0.8, "bbox": [100, 100, 200, 150]}
            ]
        }

    def process_video(self, video_path, query_params, sample_rate=5):
        return {
            "0": [{"label": "person", "confidence": 0.9, "bbox": [10, 10, 50, 50]}],
            "5": [{"label": "person", "confidence": 0.85, "bbox": [20, 20, 60, 60]}]
        }


@pytest.fixture
def pipeline():
    """Create a pipeline with mocked processors."""
    pipeline = Pipeline()

    # Mock registry and processors
    pipeline.llm_processor = MockLLMProcessor()
    pipeline.vision_processor = MockVisionProcessor()
    pipeline.media_processor = MagicMock(spec=MediaProcessor)

    # Set up media processor mock
    pipeline.media_processor.get_output_path.return_value = "/mock/output/path.jpg"
    pipeline.media_processor.is_video.return_value = False

    return pipeline


@pytest.fixture
def test_image():
    """Create a temporary test image."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        temp_path = temp.name
        # Create a minimal valid JPG
        with open(temp_path, 'wb') as f:
            f.write(
                b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x01??\x00\xff\xd9')

    return temp_path


@pytest.fixture
def test_video():
    """Create a temporary test video file (just the name, not actual content)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
        temp_path = temp.name

    return temp_path


def test_pipeline_initialization():
    """Test pipeline initialization."""
    pipeline = Pipeline()

    # Check if configuration is loaded
    assert pipeline.config is not None
    assert "llm" in pipeline.config.config
    assert "vision" in pipeline.config.config
    assert "media" in pipeline.config.config

    # Processors should be None initially
    assert pipeline.llm_processor is None
    assert pipeline.vision_processor is None
    assert pipeline.media_processor is not None


def test_process_image(pipeline, test_image):
    """Test processing an image."""
    # Process the image
    result = pipeline.process("What objects are in this image?", test_image)

    # Check result structure
    assert "query" in result
    assert "media_path" in result
    assert "media_type" in result
    assert "output_path" in result
    assert "explanation" in result
    assert "detections" in result
    assert "query_params" in result
    assert "highlighted_objects" in result

    # Check specific values
    assert result["query"] == "What objects are in this image?"
    assert result["media_path"] == test_image
    assert result["media_type"] == "image"
    assert result["output_path"] == "/mock/output/path.jpg"
    assert result["explanation"] == "I detected a person in the image."
    assert "0" in result["detections"]
    assert len(result["detections"]["0"]) == 2
    assert result["detections"]["0"][0]["label"] == "person"
    assert result["detections"]["0"][1]["label"] == "car"

    # Clean up
    os.unlink(test_image)


@patch('langvio.utils.file_utils.is_video_file')
def test_process_video(mock_is_video, pipeline, test_video):
    """Test processing a video."""
    # Mock is_video_file to return True
    mock_is_video.return_value = True
    pipeline.media_processor.is_video.return_value = True

    # Process the video
    result = pipeline.process("What objects are in this video?", test_video)

    # Check result structure
    assert "query" in result
    assert "media_path" in result
    assert "media_type" in result
    assert "output_path" in result
    assert "explanation" in result
    assert "detections" in result

    # Check specific values
    assert result["query"] == "What objects are in this video?"
    assert result["media_path"] == test_video
    assert result["media_type"] == "video"
    assert result["explanation"] == "I detected a person in the image."
    assert "0" in result["detections"]
    assert "5" in result["detections"]

    # Clean up
    os.unlink(test_video)


def test_get_visualization_config(pipeline):
    """Test getting visualization configuration based on query parameters."""
    # Test for different task types
    counting_params = {"task_type": "counting"}
    counting_config = pipeline._get_visualization_config(counting_params)
    assert counting_config["box_color"] == [255, 0, 0]  # Red for counting

    verification_params = {"task_type": "verification"}
    verification_config = pipeline._get_visualization_config(verification_params)
    assert verification_config["box_color"] == [0, 0, 255]  # Blue for verification

    tracking_params = {"task_type": "tracking"}
    tracking_config = pipeline._get_visualization_config(tracking_params)
    assert tracking_config["box_color"] == [255, 165, 0]  # Orange for tracking
    assert tracking_config["line_thickness"] == 3  # Thicker lines

    # Test with attributes
    attribute_params = {"task_type": "identification", "attributes": [{"attribute": "color", "value": "red"}]}
    attribute_config = pipeline._get_visualization_config(attribute_params)
    assert attribute_config["line_thickness"] > 2  # Increased line thickness for attributes


def test_missing_processor_error(pipeline):
    """Test error handling when processors are missing."""
    # Remove LLM processor
    pipeline.llm_processor = None

    with pytest.raises(SystemExit):
        pipeline.process("What objects are in this image?", "test.jpg")

    # Restore LLM processor, remove vision processor
    pipeline.llm_processor = MockLLMProcessor()
    pipeline.vision_processor = None

    with pytest.raises(SystemExit):
        pipeline.process("What objects are in this image?", "test.jpg")


def test_missing_file_error(pipeline):
    """Test error handling when the media file doesn't exist."""
    with pytest.raises(SystemExit):
        pipeline.process("What objects are in this image?", "nonexistent_file.jpg")