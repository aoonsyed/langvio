"""
Main test file that serves as an entry point for running all tests
"""

import os
import pytest
import importlib.util
from unittest.mock import patch


def test_imports():
    """Test basic imports of main modules."""
    # Test importing the package
    import langvio

    # Test importing key components
    from langvio import create_pipeline, registry
    from langvio.core.pipeline import Pipeline
    from langvio.llm.base import BaseLLMProcessor
    from langvio.vision.base import BaseVisionProcessor

    # Check package version
    assert hasattr(langvio, "__version__")
    assert isinstance(langvio.__version__, str)


@patch('langvio.registry')
def test_api_access(mock_registry, sample_config_path):
    """Test access to main API functions."""
    # Import the package
    import langvio

    # Set up mock registry
    mock_registry.list_llm_processors.return_value = {"default": object}
    mock_registry.list_vision_processors.return_value = {"yolo": object}
    mock_registry.get_llm_processor.return_value = object()
    mock_registry.get_vision_processor.return_value = object()

    # Test creating a pipeline
    pipeline = langvio.create_pipeline(config_path=sample_config_path)
    assert isinstance(pipeline, langvio.Pipeline)

    # Test accessing registry
    assert langvio.registry is not None


def test_package_structure():
    """Test the overall package structure."""
    # Check core modules exist
    core_modules = [
        "langvio/core/pipeline.py",
        "langvio/core/registry.py",
        "langvio/core/base.py",
        "langvio/llm/base.py",
        "langvio/vision/base.py",
        "langvio/media/processor.py",
        "langvio/utils/file_utils.py",
        "langvio/utils/logging.py",
        "langvio/utils/llm_utils.py",
        "langvio/utils/vision_utils.py"
    ]

    for module_path in core_modules:
        # Get the absolute path in relation to the package
        if importlib.util.find_spec("langvio"):
            pkg_path = os.path.dirname(importlib.util.find_spec("langvio").origin)
            parent_path = os.path.dirname(pkg_path)
            absolute_path = os.path.join(parent_path, module_path)

            # Check only if we're in the correct environment (e.g., not in CI where paths might differ)
            if os.path.exists(parent_path):
                assert os.path.exists(absolute_path), f"Missing module: {module_path}"


def test_cli_module():
    """Test that the CLI module is available."""
    # Check if CLI module exists
    cli_spec = importlib.util.find_spec("langvio.cli")
    assert cli_spec is not None, "CLI module not found"

    # Check if the main function exists in the CLI module
    cli_module = importlib.util.import_module("langvio.cli")
    assert hasattr(cli_module, "main"), "main function not found in CLI module"