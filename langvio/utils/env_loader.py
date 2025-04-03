"""
Utilities for loading environment variables from .env files
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


def load_dotenv(dotenv_path: Optional[str] = None, override: bool = False) -> bool:
    """
    Load environment variables from a .env file.

    Args:
        dotenv_path: Path to the .env file. If None, searches in current directory.
        override: Whether to override existing environment variables.

    Returns:
        True if the .env file was loaded successfully, False otherwise.
    """
    try:
        # Try to import python-dotenv
        try:
            from dotenv import load_dotenv as dotenv_load

            # If a specific path is provided, use it
            if dotenv_path:
                return dotenv_load(dotenv_path, override=override)

            # Otherwise try to find .env file in common locations
            dotenv_paths = [
                ".env",
                Path.home() / ".env",
                Path(__file__).parent.parent / ".env",
            ]

            for path in dotenv_paths:
                if os.path.isfile(path):
                    return dotenv_load(path, override=override)

            logger.warning("No .env file found in common locations.")
            return False

        except ImportError:
            # If python-dotenv is not installed, try to load manually
            logger.warning("python-dotenv not installed, trying manual .env loading")
            return _manual_load_dotenv(dotenv_path, override)

    except Exception as e:
        logger.warning(f"Error loading .env file: {e}")
        return False


def _manual_load_dotenv(dotenv_path: Optional[str] = None, override: bool = False) -> bool:
    """
    Manually load environment variables from a .env file without dependencies.

    Args:
        dotenv_path: Path to the .env file. If None, tries to find it in current directory.
        override: Whether to override existing environment variables.

    Returns:
        True if the .env file was loaded successfully, False otherwise.
    """
    # Determine path to .env file
    if dotenv_path is None:
        dotenv_path = ".env"
        if not os.path.isfile(dotenv_path):
            # Try home directory
            home_env = os.path.join(str(Path.home()), ".env")
            if os.path.isfile(home_env):
                dotenv_path = home_env

    # Check if file exists
    if not os.path.isfile(dotenv_path):
        logger.warning(f"No .env file found at {dotenv_path}")
        return False

    # Parse .env file
    try:
        with open(dotenv_path, 'r') as f:
            lines = f.readlines()

        # Process each line
        for line in lines:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse key-value pair
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                        (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]

                # Set environment variable if not already set or override is True
                if override or key not in os.environ:
                    os.environ[key] = value

        return True

    except Exception as e:
        logger.warning(f"Error parsing .env file: {e}")
        return False


def check_required_keys(required_keys: List[str]) -> Dict[str, bool]:
    """
    Check if all required API keys are set in environment variables.

    Args:
        required_keys: List of required API key environment variable names.

    Returns:
        Dictionary mapping key names to boolean indicating if they're set.
    """
    return {key: key in os.environ and bool(os.environ[key]) for key in required_keys}