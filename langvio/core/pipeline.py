"""
Enhanced core pipeline for connecting LLMs with vision models
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2

from langvio.config import Config
from langvio.llm.base import BaseLLMProcessor
from langvio.media.processor import MediaProcessor
from langvio.utils.file_utils import is_image_file, is_video_file
from langvio.utils.logging import setup_logging
from langvio.vision.base import BaseVisionProcessor


class Pipeline:
    """Enhanced main pipeline for processing queries with LLMs and vision models"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pipeline.

        Args:
            config_path: Path to configuration file
        """
        # Initialize configuration
        self.config = Config(config_path)

        # Set up logging
        setup_logging(self.config.get_logging_config())
        self.logger = logging.getLogger(__name__)

        # Initialize processors
        self.llm_processor = None
        self.vision_processor = None
        self.media_processor = MediaProcessor(self.config.get_media_config())

        self.logger.info("Enhanced Pipeline initialized")

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file
        """
        self.config.load_config(config_path)
        self.logger.info(f"Loaded configuration from {config_path}")

        # Reinitialize processors with new config
        if self.llm_processor:
            self.set_llm_processor(self.llm_processor.name)

        if self.vision_processor:
            self.set_vision_processor(self.vision_processor.name)

        # Update media processor
        self.media_processor.update_config(self.config.get_media_config())

    def set_llm_processor(self, processor_name: str) -> None:
        """
        Set the LLM processor.

        Args:
            processor_name: Name of the processor to use
        """
        from langvio import registry

        self.logger.info(f"Setting LLM processor to {processor_name}")

        # Get processor config
        processor_config = self.config.get_llm_config(processor_name)

        # Check if the requested processor is available
        if processor_name not in registry.list_llm_processors():
            error_msg = (
                f"ERROR: LLM processor '{processor_name}' not found. "
                "You may need to install additional dependencies:\n"
                "- For OpenAI: pip install langvio[openai]\n"
                "- For Google Gemini: pip install langvio[google]\n"
                "- For all providers: pip install langvio[all-llm]"
            )
            self.logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)

        # Create processor
        try:
            self.llm_processor = registry.get_llm_processor(
                processor_name, **processor_config
            )

            # Explicitly initialize the processor
            self.llm_processor.initialize()

        except Exception as e:
            error_msg = (
                f"ERROR: Failed to initialize LLM processor '{processor_name}': {e}"
            )
            self.logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)

    def set_vision_processor(self, processor_name: str) -> None:
        """
        Set the vision processor.

        Args:
            processor_name: Name of the processor to use
        """
        from langvio import registry

        self.logger.info(f"Setting vision processor to {processor_name}")

        # Get processor config
        processor_config = self.config.get_vision_config(processor_name)

        # Check if the requested processor is available
        if processor_name not in registry.list_vision_processors():
            error_msg = f"ERROR: Vision processor '{processor_name}' not found."
            self.logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)

        # Create processor
        try:
            self.vision_processor = registry.get_vision_processor(
                processor_name, **processor_config
            )
        except Exception as e:
            error_msg = (
                f"ERROR: Failed to initialize vision processor '{processor_name}': {e}"
            )
            self.logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)

    def process(self, query: str, media_path: str) -> Dict[str, Any]:
        """
        Process a query on media with enhanced capabilities.
        Modified to pass all objects to the LLM before filtering for visualization.

        Args:
            query: Natural language query
            media_path: Path to media file (image or video)

        Returns:
            Dictionary with results

        Raises:
            ValueError: If processors are not set or media file doesn't exist
        """
        self.logger.info(f"Processing query: {query}")

        # Check if processors are set
        if not self.llm_processor:
            error_msg = "ERROR: LLM processor not set"
            self.logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)

        if not self.vision_processor:
            error_msg = "ERROR: Vision processor not set"
            self.logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)

        # Check if media file exists
        if not os.path.exists(media_path):
            error_msg = f"ERROR: Media file not found: {media_path}"
            self.logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)

        # Check media type
        is_video = is_video_file(media_path)
        media_type = "video" if is_video else "image"

        # Process query with LLM
        query_params = self.llm_processor.parse_query(query)
        self.logger.info(f"Parsed query params: {query_params}")

        # Run detection with vision processor - WITHOUT INITIAL FILTERING
        if is_video:
            # For video processing, check if we need to adjust sample rate based on task
            sample_rate = 5  # Default
            if query_params.get("task_type") in ["tracking", "activity"]:
                # Use a more frequent sampling for tracking and activity detection
                sample_rate = 2

            # Get all detections without filtering
            all_detections = self.vision_processor.process_video(
                media_path, query_params, sample_rate
            )
        else:
            # Get all detections without filtering
            all_detections = self.vision_processor.process_image(
                media_path, query_params
            )

            # Generate explanation using ALL detected objects
        explanation = self.llm_processor.generate_explanation(query, all_detections)

        # Get highlighted objects from the LLM processor
        highlighted_objects = self.llm_processor.get_highlighted_objects()

        # Create visualization with highlighted objects
        output_path = self._create_visualization(
            media_path, all_detections, highlighted_objects, query_params, is_video
        )

        # Prepare result
        result = {
            "query": query,
            "media_path": media_path,
            "media_type": media_type,
            "output_path": output_path,
            "explanation": explanation,
            "detections": all_detections,
            "query_params": query_params,
            "highlighted_objects": highlighted_objects,
        }

        self.logger.info(f"Processed query successfully")
        return result

    def _get_visualization_config(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get visualization configuration based on query parameters.

        Args:
            query_params: Query parameters from LLM processor

        Returns:
            Visualization configuration parameters
        """
        # Get default visualization config
        viz_config = self.config.config["media"]["visualization"].copy()

        # Customize based on task type
        task_type = query_params.get("task_type", "identification")

        if task_type == "counting":
            # For counting tasks, use a different color
            viz_config["box_color"] = [255, 0, 0]  # Red for counting

        elif task_type == "verification":
            # For verification tasks, use a different color
            viz_config["box_color"] = [0, 0, 255]  # Blue for verification

        elif task_type in ["tracking", "activity"]:
            # For tracking/activity tasks, use a more visible color
            viz_config["box_color"] = [255, 165, 0]  # Orange for tracking/activity
            viz_config["line_thickness"] = 3  # Thicker lines

        # If specific attributes were requested, adjust the visualization
        if query_params.get("attributes"):
            # If looking for specific attributes, highlight them more
            viz_config["line_thickness"] += 1

        return viz_config

    def _create_visualization(
        self,
        media_path: str,
        all_detections: Dict[str, List[Dict[str, Any]]],
        highlighted_objects: List[Dict[str, Any]],
        query_params: Dict[str, Any],
        is_video: bool,
    ) -> str:
        """
        Create visualization with highlighted objects and return the output path.

        Args:
            media_path: Path to input media
            all_detections: All detection results
            highlighted_objects: Objects to highlight
            query_params: Query parameters
            is_video: Whether the media is a video

        Returns:
            Path to the output visualization
        """
        # Import vision utils
        from langvio.utils.vision_utils import (
            create_visualization_detections_for_image,
            create_visualization_detections_for_video)

        # Generate output path
        output_path = self.media_processor.get_output_path(media_path)

        # Get visualization config
        visualization_config = self._get_visualization_config(query_params)

        if is_video:
            # Create visualization detections for video
            visualization_detections = create_visualization_detections_for_video(
                all_detections, highlighted_objects
            )

            # Visualize video
            self.media_processor.visualize_video(
                media_path,
                output_path,
                visualization_detections,
                **visualization_config,
            )
        else:
            # Create visualization detections for image
            visualization_detections = create_visualization_detections_for_image(
                highlighted_objects
            )

            # Visualize image
            self.media_processor.visualize_image(
                media_path,
                output_path,
                visualization_detections,
                **visualization_config,
            )

        return output_path
