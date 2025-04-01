"""
Utility functions for YOLO processors
"""

from typing import List, Dict, Any


def extract_detections(results) -> List[Dict[str, Any]]:
    """
    Extract detections from YOLO results.

    Args:
        results: Raw YOLO results

    Returns:
        List of detection dictionaries
    """
    detections = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = result.names[cls_id]

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "class_id": cls_id
            })

    return detections


def filter_by_confidence(detections: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    """
    Filter detections by confidence threshold.

    Args:
        detections: List of detection dictionaries
        threshold: Confidence threshold

    Returns:
        Filtered list of detections
    """
    return [det for det in detections if det["confidence"] >= threshold]


def intersection_over_union(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]

    Returns:
        IoU value
    """
    # Box coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0  # No intersection

    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou


def non_max_suppression(detections: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.

    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for suppression

    Returns:
        Filtered list of detections
    """
    # Sort detections by confidence
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

    # Initialize list of selected detections
    selected = []

    # Iterate through detections
    while detections:
        # Add detection with highest confidence to selected list
        selected.append(detections[0])

        # Remove selected detection from list
        current_box = detections[0]["bbox"]
        detections = detections[1:]

        # Filter remaining detections
        filtered_detections = []

        for det in detections:
            # Calculate IoU with selected detection
            iou = intersection_over_union(current_box, det["bbox"])

            # Keep detection if IoU is below threshold
            if iou < iou_threshold:
                filtered_detections.append(det)

        detections = filtered_detections

    return selected