# Langvio Test Suite

This directory contains tests for the Langvio library. These tests focus on basic functionality and do not aim for high coverage or complex cases.

## Test Structure

The test suite is organized as follows:

- `conftest.py`: Common fixtures used across tests
- `test_main.py`: Tests for basic imports and package structure
- `test_config.py`: Tests for configuration loading and management
- `test_registry.py`: Tests for model registry functionality
- `test_file_utils.py`: Tests for file utility functions
- `test_llm_utils.py`: Tests for LLM utility functions
- `test_color_detector.py`: Tests for color detection module
- `test_pipeline.py`: Tests for the main pipeline functionality
- `test_create_pipeline.py`: Tests for the pipeline creation function

## Running Tests

To run the tests, use pytest:

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_config.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=langvio
```

## Test Coverage

These tests cover basic functionality of the main components:

1. **Configuration Management**
   - Loading default configuration
   - Loading custom configuration
   - Getting specific configurations
   - Saving configuration to file
   - Error handling for invalid configurations

2. **Model Registry**
   - Registering LLM processors
   - Registering vision processors
   - Retrieving processors
   - Error handling for non-existent processors

3. **File Utilities**
   - Directory management
   - File extension handling
   - Image/video file detection
   - Creating temporary copies
   - Finding files in directories

4. **LLM Utilities**
   - Detection indexing
   - Object ID extraction
   - Retrieving objects by IDs
   - Parsing LLM responses

5. **Color Detection**
   - Detecting solid colors
   - Detecting multicolors
   - Getting color profiles
   - Finding objects by color

6. **Pipeline Functionality**
   - Pipeline initialization
   - Processing images
   - Processing videos
   - Error handling

7. **Pipeline Creation**
   - Creating pipelines with default settings
   - Creating pipelines with custom configuration
   - Creating pipelines with specific processors
   - Error handling for missing processors

## Adding Tests

When adding new tests:

1. Create a new test file if testing a new component
2. Use fixtures from `conftest.py` where appropriate
3. Mock external dependencies and hardware-dependent features
4. Focus on testing behavior, not implementation details
5. Keep tests independent and idempotent



