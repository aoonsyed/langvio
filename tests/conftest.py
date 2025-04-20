"""
Common pytest fixtures for Langvio tests
"""

import os
import pytest
import tempfile
import numpy as np
import cv2


@pytest.fixture(scope="session")
def test_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tempdir:
        # Create subdirectories
        os.makedirs(os.path.join(tempdir, "output"), exist_ok=True)
        os.makedirs(os.path.join(tempdir, "temp"), exist_ok=True)

        yield tempdir


@pytest.fixture
def sample_image_path(test_dir):
    """Create a sample image for testing."""
    # Create a simple test image (100x100 RGB)
    image = np.ones((100, 100, 3), dtype=np.uint8) * 200  # Light gray

    # Add a red rectangle
    image[20:50, 30:70, 0] = 0  # B channel
    image[20:50, 30:70, 1] = 0  # G channel
    image[20:50, 30:70, 2] = 255  # R channel

    # Add a blue circle
    cv2.circle(image, (70, 70), 15, (255, 0, 0), -1)  # BGR format

    # Save the image
    image_path = os.path.join(test_dir, "sample_image.jpeg")
    cv2.imwrite(image_path, image)

    yield image_path

    # Clean up
    if os.path.exists(image_path):
        os.unlink(image_path)


@pytest.fixture
def mock_video_path(test_dir):
    """Create a mock video file for testing."""
    # For testing purposes, we only need the file path, not actual content
    video_path = os.path.join(test_dir, "sample_video.mp4")

    # Create an empty file
    with open(video_path, "wb") as f:
        f.write(b"dummy video content")

    yield video_path

    # Clean up
    if os.path.exists(video_path):
        os.unlink(video_path)


@pytest.fixture
def sample_config_dict():
    """Return a sample configuration dictionary."""
    return {
        "llm": {
            "default": "test-llm",
            "models": {
                "test-llm": {
                    "model_name": "test-model",
                    "model_kwargs": {"temperature": 0.5},
                }
            },
        },
        "vision": {
            "default": "test-vision",
            "models": {
                "test-vision": {
                    "type": "test",
                    "model_path": "test-model.pt",
                    "confidence": 0.3,
                }
            },
        },
        "media": {
            "output_dir": "./test-output",
            "temp_dir": "./test-temp",
            "visualization": {
                "box_color": [255, 0, 0],
                "text_color": [0, 0, 0],
                "line_thickness": 3,
            },
        },
        "logging": {"level": "DEBUG", "file": "test.log"},
    }


@pytest.fixture
def sample_config_path(sample_config_dict):
    """Create a sample configuration file."""
    import yaml

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as temp:
        temp_path = temp.name
        yaml.dump(sample_config_dict, temp)

    yield temp_path

    # Clean up
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_detections():
    """Return sample detection results."""
    return {
        "0": [
            {
                "label": "person",
                "confidence": 0.92,
                "bbox": [10, 20, 110, 220],
                "class_id": 0,
                "center": (60, 120),
                "dimensions": (100, 200),
                "area": 20000,
                "attributes": {"color": "red", "size": "medium"},
            },
            {
                "label": "car",
                "confidence": 0.85,
                "bbox": [200, 300, 400, 400],
                "class_id": 2,
                "center": (300, 350),
                "dimensions": (200, 100),
                "area": 20000,
                "attributes": {"color": "blue", "size": "large"},
            },
        ]
    }


@pytest.fixture
def sample_video_detections():
    """Return sample video detection results."""
    return {
        "0": [
            {
                "label": "person",
                "confidence": 0.92,
                "bbox": [10, 20, 110, 220],
                "track_id": 1,
                "activities": ["walking"],
            }
        ],
        "5": [
            {
                "label": "person",
                "confidence": 0.90,
                "bbox": [15, 25, 115, 225],
                "track_id": 1,
                "activities": ["walking"],
            }
        ],
        "10": [
            {
                "label": "person",
                "confidence": 0.88,
                "bbox": [20, 30, 120, 230],
                "track_id": 1,
                "activities": ["running"],
            },
            {
                "label": "car",
                "confidence": 0.85,
                "bbox": [200, 300, 400, 400],
                "track_id": 2,
                "activities": ["moving"],
            },
        ],
    }
