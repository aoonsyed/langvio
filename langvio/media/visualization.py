"""
Visualization utilities for media
"""

from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np


def draw_detections_on_image(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    box_color: Union[Tuple[int, int, int], List[int]] = (0, 255, 0),
    text_color: Union[Tuple[int, int, int], List[int]] = (255, 255, 255),
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw detections on an image.

    Args:
        image: Input image as numpy array
        detections: List of detection dictionaries
        box_color: Color for bounding boxes (BGR)
        text_color: Color for text (BGR)
        line_thickness: Thickness of bounding box lines

    Returns:
        Image with detections drawn
    """
    # Create a copy of the image
    output_image = image.copy()

    # Draw each detection
    for det in detections:
        # Extract bounding box
        x1, y1, x2, y2 = det["bbox"]

        # Create label with confidence
        label = f"{det['label']} {det['confidence']:.2f}"

        # Draw bounding box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), box_color, line_thickness)

        # Get text size
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        # Draw text background
        cv2.rectangle(
            output_image,
            (x1, y1 - text_size[1] - 5),
            (x1 + text_size[0], y1),
            box_color,
            -1,
        )

        # Draw text
        cv2.putText(
            output_image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            2,
        )

    return output_image


def draw_detections_on_video(
    input_path: str,
    output_path: str,
    frame_detections: Dict[str, List[Dict[str, Any]]],
    box_color: Union[Tuple[int, int, int], List[int]] = (0, 255, 0),
    text_color: Union[Tuple[int, int, int], List[int]] = (255, 255, 255),
    line_thickness: int = 2,
) -> None:
    """
    Draw detections on a video.

    Args:
        input_path: Path to input video
        output_path: Path to save output video
        frame_detections: Dictionary mapping frame indices to detections
        box_color: Color for bounding boxes (BGR)
        text_color: Color for text (BGR)
        line_thickness: Thickness of bounding box lines
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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process frames
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if we have detections for this frame
        if str(frame_idx) in frame_detections:
            # Draw detections
            frame = draw_detections_on_image(
                frame,
                frame_detections[str(frame_idx)],
                box_color,
                text_color,
                line_thickness,
            )

        # Write frame
        writer.write(frame)
        frame_idx += 1

    # Clean up
    cap.release()
    writer.release()
