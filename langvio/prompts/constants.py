"""
Constants for langvio package
"""

# Task types that the system can handle
TASK_TYPES = [
    "identification",  # Basic object detection
    "counting",  # Counting specific objects
    "verification",  # Verifying existence of objects
    "analysis",  # Detailed analysis with attributes and relationships
    "tracking",  # For tracking objects across video frames
    "activity",  # For detecting activities/actions
]

# Common attributes that can be detected
VISUAL_ATTRIBUTES = [
    "color",
    "size",
    "shape",
    "posture",
    "state",  # open/closed, on/off
    "texture",
    "material",
]

# Common spatial relationships between objects
SPATIAL_RELATIONS = [
    "above",
    "below",
    "next_to",
    "inside",
    "outside",
    "on_top_of",
    "under",
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "between",
    "near",
    "far",
]

# Common object activities/actions (for videos)
ACTIVITIES = [
    "walking",
    "running",
    "sitting",
    "standing",
    "jumping",
    "eating",
    "drinking",
    "talking",
    "moving",
    "stationary",
]

# Common COCO dataset object categories
COMMON_OBJECTS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Default detection confidence threshold
DEFAULT_CONFIDENCE_THRESHOLD = 0.25

# Default IoU threshold for NMS
DEFAULT_IOU_THRESHOLD = 0.5

# Default sample rate for video processing (every N frames)
DEFAULT_VIDEO_SAMPLE_RATE = 5
