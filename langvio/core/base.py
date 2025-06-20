"""
Base classes for langvio components
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Processor(ABC):
    """Base class for all processors in langvio"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize processor.

        Args:
            name: Processor name
            config: Configuration parameters
        """
        self.name = name
        self.config = config or {}

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the processor with its configuration.

        Returns:
            True if initialization was successful
        """

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> "Processor":
        """
        Create a processor from configuration.

        Args:
            name: Processor name
            config: Configuration parameters

        Returns:
            Initialized processor
        """
        processor = cls(name, config)
        processor.initialize()
        return processor


class Model(ABC):
    """Base class for all models in langvio"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model.

        Args:
            name: Model name
            config: Configuration parameters
        """
        self.name = name
        self.config = config or {}
        self.initialized = False

    @abstractmethod
    def load(self) -> bool:
        """
        Load the model.

        Returns:
            True if model was loaded successfully
        """

    @abstractmethod
    def unload(self) -> bool:
        """
        Unload the model to free resources.

        Returns:
            True if model was unloaded successfully
        """
