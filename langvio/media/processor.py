"""
Enhanced media processing utilities
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

import cv2
import numpy as np


class MediaProcessor:
    """Enhanced processor for handling media files (images and videos)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize media processor.

        Args:
            config: Configuration parameters
        """
        self.config = config or {
            "output_dir": "./output",
            "temp_dir": "./temp",
            "visualization": {
                "box_color": [0, 255, 0],
                "text_color": [255, 255, 255],
                "line_thickness": 2,
                "show_attributes": True,
                "show_confidence": True
            }
        }

        self.logger = logging.getLogger(__name__)

        # Create output and temp directories
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs(self.config["temp_dir"], exist_ok=True)

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update configuration parameters.

        Args:
            config: New configuration parameters
        """
        self.config.update(config)

        # Ensure directories exist
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs(self.config["temp_dir"], exist_ok=True)

    def is_video(self, file_path: str) -> bool:
        """
        Check if a file is a video based on extension.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is a video
        """
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        _, ext = os.path.splitext(file_path.lower())
        return ext in video_extensions

    def get_output_path(self, input_path: str, suffix: str = "_processed") -> str:
        """
        Generate an output path for processed media.

        Args:
            input_path: Path to the input file
            suffix: Suffix to add to the filename

        Returns:
            Output path
        """
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}{suffix}{ext}"

        return os.path.join(self.config["output_dir"], output_filename)

    def visualize_image(self, image_path: str, output_path: str,
                        detections: List[Dict[str, Any]],
                        box_color: Optional[List[int]] = None,
                        text_color: Optional[List[int]] = None,
                        line_thickness: Optional[int] = None,
                        show_attributes: Optional[bool] = None,
                        show_confidence: Optional[bool] = None) -> None:
        """
        Enhanced visualization of detections on an image.

        Args:
            image_path: Path to the input image
            output_path: Path to save the output image
            detections: List of detection dictionaries
            box_color: Color for bounding boxes (BGR)
            text_color: Color for text (BGR)
            line_thickness: Thickness of bounding box lines
            show_attributes: Whether to display attribute information
            show_confidence: Whether to display confidence scores
        """
        self.logger.info(f"Visualizing {len(detections)} detections on image: {image_path}")

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Get visualization config (use provided params or defaults from config)
            viz_config = self.config["visualization"]

            # Override with provided parameters if any
            if box_color is not None:
                viz_config["box_color"] = box_color
            if text_color is not None:
                viz_config["text_color"] = text_color
            if line_thickness is not None:
                viz_config["line_thickness"] = line_thickness
            if show_attributes is not None:
                viz_config["show_attributes"] = show_attributes
            if show_confidence is not None:
                viz_config["show_confidence"] = show_confidence

            # Draw detections with enhanced visualization
            image_with_detections = self._draw_detections_on_image(
                image,
                detections,
                box_color=viz_config["box_color"],
                text_color=viz_config["text_color"],
                line_thickness=viz_config["line_thickness"],
                show_attributes=viz_config.get("show_attributes", True),
                show_confidence=viz_config.get("show_confidence", True)
            )

            # Save output
            cv2.imwrite(output_path, image_with_detections)
            self.logger.info(f"Saved visualized image to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error visualizing image: {e}")

    def visualize_video(self, video_path: str, output_path: str,
                        frame_detections: Dict[str, List[Dict[str, Any]]],
                        box_color: Optional[List[int]] = None,
                        text_color: Optional[List[int]] = None,
                        line_thickness: Optional[int] = None,
                        show_attributes: Optional[bool] = None,
                        show_confidence: Optional[bool] = None) -> None:
        """
        Enhanced visualization of detections on a video.

        Args:
            video_path: Path to the input video
            output_path: Path to save the output video
            frame_detections: Dictionary mapping frame indices to detections
            box_color: Color for bounding boxes (BGR)
            text_color: Color for text (BGR)
            line_thickness: Thickness of bounding box lines
            show_attributes: Whether to display attribute information
            show_confidence: Whether to display confidence scores
        """
        self.logger.info(f"Visualizing detections on video: {video_path}")

        try:
            # Get visualization config (use provided params or defaults from config)
            viz_config = self.config["visualization"]

            # Override with provided parameters if any
            if box_color is not None:
                viz_config["box_color"] = box_color
            if text_color is not None:
                viz_config["text_color"] = text_color
            if line_thickness is not None:
                viz_config["line_thickness"] = line_thickness
            if show_attributes is not None:
                viz_config["show_attributes"] = show_attributes
            if show_confidence is not None:
                viz_config["show_confidence"] = show_confidence

            # Enhanced drawing on video
            self._draw_detections_on_video(
                video_path,
                output_path,
                frame_detections,
                box_color=viz_config["box_color"],
                text_color=viz_config["text_color"],
                line_thickness=viz_config["line_thickness"],
                show_attributes=viz_config.get("show_attributes", True),
                show_confidence=viz_config.get("show_confidence", True)
            )

            self.logger.info(f"Saved visualized video to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error visualizing video: {e}")

    def _draw_detections_on_image(self,
                                 image: np.ndarray,
                                 detections: List[Dict[str, Any]],
                                 box_color: Union[Tuple[int, int, int], List[int]] = (0, 255, 0),
                                 text_color: Union[Tuple[int, int, int], List[int]] = (255, 255, 255),
                                 line_thickness: int = 2,
                                 show_attributes: bool = True,
                                 show_confidence: bool = True) -> np.ndarray:
        """
        Enhanced method to draw detections on an image with attributes and activities.

        Args:
            image: Input image as numpy array
            detections: List of detection dictionaries
            box_color: Color for bounding boxes (BGR)
            text_color: Color for text (BGR)
            line_thickness: Thickness of bounding box lines
            show_attributes: Whether to display attribute information
            show_confidence: Whether to display confidence scores

        Returns:
            Image with detections drawn
        """
        # Create a copy of the image
        output_image = image.copy()

        # Draw each detection
        for det in detections:
            # Extract bounding box
            x1, y1, x2, y2 = det["bbox"]

            # Create label based on configuration
            label_parts = [det['label']]

            # Add confidence if requested
            if show_confidence:
                label_parts.append(f"{det['confidence']:.2f}")

            # Add attributes if requested and present
            if show_attributes and "attributes" in det and det["attributes"]:
                for attr_name, attr_value in det["attributes"].items():
                    label_parts.append(f"{attr_name}:{attr_value}")

            # Add activities if present
            if "activities" in det and det["activities"]:
                activities = ", ".join(det["activities"])
                label_parts.append(f"[{activities}]")

            # Combine into label
            label = " | ".join(label_parts)

            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), box_color, line_thickness)

            # Calculate text size for proper background
            font_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label, font, font_scale, 2)[0]

            # Draw text background
            cv2.rectangle(output_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), box_color, -1)

            # Draw text
            cv2.putText(output_image, label, (x1, y1 - 5), font, font_scale, text_color, 2)

            # Draw relationship lines if available and showing attributes
            if show_attributes and "relationships" in det:
                for rel in det["relationships"]:
                    # Find the related object in the current detections
                    for rel_det_idx, rel_det in enumerate(detections):
                        if rel_det_idx == rel.get("object_id"):
                            # Draw a line between the centers of the objects
                            center1 = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            rel_box = rel_det["bbox"]
                            center2 = (int((rel_box[0] + rel_box[2]) / 2), int((rel_box[1] + rel_box[3]) / 2))

                            # Use different line styles for different relations
                            if "near" in rel.get("relations", []):
                                # Dashed line for "near"
                                self._draw_dashed_line(output_image, center1, center2, box_color,
                                                     thickness=line_thickness)
                            else:
                                # Solid line for other relations
                                cv2.line(output_image, center1, center2, box_color,
                                        thickness=max(1, line_thickness - 1))

                            # Draw a small text indicating the relation
                            if rel.get("relations"):
                                mid_point = (int((center1[0] + center2[0]) / 2), int((center1[1] + center2[1]) / 2))
                                rel_text = "+".join(rel["relations"][:2])  # Show at most 2 relations
                                cv2.putText(output_image, rel_text, mid_point, font, 0.4, text_color, 1)
                            break

        return output_image

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=1, gap=5):
        """
        Draw a dashed line on an image.

        Args:
            img: Image to draw on
            pt1: First point
            pt2: Second point
            color: Line color
            thickness: Line thickness
            gap: Gap between dashes
        """
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r))
            y = int((pt1[1] * (1 - r) + pt2[1] * r))
            pts.append((x, y))

        for i in range(len(pts) - 1):
            if i % 2 == 0:
                cv2.line(img, pts[i], pts[i + 1], color, thickness)

    def _draw_detections_on_video(self,
                                 input_path: str,
                                 output_path: str,
                                 frame_detections: Dict[str, List[Dict[str, Any]]],
                                 box_color: Union[Tuple[int, int, int], List[int]] = (0, 255, 0),
                                 text_color: Union[Tuple[int, int, int], List[int]] = (255, 255, 255),
                                 line_thickness: int = 2,
                                 show_attributes: bool = True,
                                 show_confidence: bool = True) -> None:
        """
        Enhanced method to draw detections on a video with tracking visualization.

        Args:
            input_path: Path to input video
            output_path: Path to save output video
            frame_detections: Dictionary mapping frame indices to detections
            box_color: Color for bounding boxes (BGR)
            text_color: Color for text (BGR)
            line_thickness: Thickness of bounding box lines
            show_attributes: Whether to display attribute information
            show_confidence: Whether to display confidence scores
        """
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {input_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # For tracking visualization - keep track of past positions
        tracks = {}  # Dictionary mapping track_id to list of past positions

        # Process frames
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Check if we have detections for this frame
            frame_key = str(frame_idx)
            if frame_key in frame_detections:
                # Get current detections
                detections = frame_detections[frame_key]

                # Update tracks for visualization
                for det in detections:
                    if "track_id" in det:
                        track_id = det["track_id"]
                        # Get center of bounding box
                        bbox = det["bbox"]
                        center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

                        # Add to track history
                        if track_id not in tracks:
                            tracks[track_id] = []

                        # Keep only last 30 positions
                        if len(tracks[track_id]) > 30:
                            tracks[track_id] = tracks[track_id][-30:]

                        tracks[track_id].append(center)

                # Draw trajectory lines for tracked objects
                for track_id, positions in tracks.items():
                    if len(positions) > 1:
                        # Generate unique color for this track
                        track_color = self._get_color_for_id(track_id)

                        # Draw line connecting positions
                        for i in range(len(positions) - 1):
                            cv2.line(frame, positions[i], positions[i + 1], track_color,
                                    thickness=max(1, line_thickness - 1))

                # Draw detections
                frame = self._draw_detections_on_image(
                    frame,
                    detections,
                    box_color,
                    text_color,
                    line_thickness,
                    show_attributes,
                    show_confidence
                )

            # Write frame
            writer.write(frame)
            frame_idx += 1

        # Clean up
        cap.release()
        writer.release()

    def _get_color_for_id(self, track_id: int) -> Tuple[int, int, int]:
        """
        Generate a consistent color for a given track ID.

        Args:
            track_id: Track identifier

        Returns:
            BGR color tuple
        """
        # Use the track_id to generate repeatable colors
        hue = (track_id * 137 % 360) / 360.0  # Use prime number to distribute colors
        sat = 0.7 + (track_id % 3) * 0.1      # Vary saturation slightly
        val = 0.8 + (track_id % 2) * 0.2      # Vary value slightly

        # Convert HSV to RGB then to BGR
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)

        return (b, g, r)  # Return BGR for OpenCV