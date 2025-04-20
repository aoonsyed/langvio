import os
import pytest
import tempfile
import yaml
from langvio.config import Config


def test_default_config_loading():
    """Test loading the default configuration."""
    config = Config()

    # Check if basic sections exist
    assert "llm" in config.config
    assert "vision" in config.config
    assert "media" in config.config
    assert "logging" in config.config

    # Check if default LLM is set
    assert "default" in config.config["llm"]
    assert "models" in config.config["llm"]

    # Check if default vision processor is set
    assert "default" in config.config["vision"]
    assert "models" in config.config["vision"]


def test_custom_config_loading():
    """Test loading a custom configuration file."""
    # Create a temporary config file
    test_config = {
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

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as temp:
        temp_path = temp.name
        yaml.dump(test_config, temp)

    try:
        # Load the custom config
        config = Config(temp_path)

        # Check if custom values are loaded
        assert config.config["llm"]["default"] == "test-llm"
        assert config.config["vision"]["default"] == "test-vision"
        assert config.config["media"]["output_dir"] == "./test-output"
        assert config.config["logging"]["level"] == "DEBUG"

        # Check model specifics
        assert config.config["llm"]["models"]["test-llm"]["model_name"] == "test-model"
        assert config.config["vision"]["models"]["test-vision"]["confidence"] == 0.3
        assert config.config["media"]["visualization"]["box_color"] == [255, 0, 0]
    finally:
        # Clean up
        os.unlink(temp_path)


def test_get_specific_configs():
    """Test getting specific configurations."""
    # Create config with custom settings
    test_config = {
        "llm": {
            "default": "default-llm",
            "models": {
                "default-llm": {"model_name": "default-model"},
                "other-llm": {"model_name": "other-model"},
            },
        },
        "vision": {
            "default": "default-vision",
            "models": {
                "default-vision": {"type": "default-type"},
                "other-vision": {"type": "other-type"},
            },
        },
    }

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as temp:
        temp_path = temp.name
        yaml.dump(test_config, temp)

    try:
        config = Config(temp_path)

        # Test getting default LLM config
        llm_config = config.get_llm_config()
        assert llm_config["model_name"] == "default-model"

        # Test getting specific LLM config
        llm_config = config.get_llm_config("other-llm")
        assert llm_config["model_name"] == "other-model"

        # Test getting default vision config
        vision_config = config.get_vision_config()
        assert vision_config["type"] == "default-type"

        # Test getting specific vision config
        vision_config = config.get_vision_config("other-vision")
        assert vision_config["type"] == "other-type"
    finally:
        # Clean up
        os.unlink(temp_path)


def test_save_config():
    """Test saving configuration to a file."""
    config = Config()

    # Modify the config
    config.config["test_section"] = {"test_key": "test_value"}

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp:
        temp_path = temp.name

    try:
        config.save_config(temp_path)

        # Load the saved config and check if modifications are there
        saved_config = Config(temp_path)
        assert "test_section" in saved_config.config
        assert saved_config.config["test_section"]["test_key"] == "test_value"
    finally:
        # Clean up
        os.unlink(temp_path)


def test_config_error_handling():
    """Test error handling when loading invalid configuration."""
    # Create a temporary invalid config file
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp:
        temp_path = temp.name
        temp.write(b"invalid: yaml: :")

    try:
        # Loading invalid config should raise an error
        with pytest.raises(ValueError):
            Config(temp_path)
    finally:
        # Clean up
        os.unlink(temp_path)
