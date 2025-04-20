from langvio.utils.llm_utils import (
    extract_object_ids,
    get_objects_by_ids,
    index_detections,
    parse_explanation_response,
)


def test_index_detections():
    """Test the index_detections function."""
    # Create sample detections
    detections = {
        "0": [
            {"label": "person", "confidence": 0.9},
            {"label": "car", "confidence": 0.8},
        ],
        "10": [{"label": "person", "confidence": 0.85}],
    }

    # Call the function
    indexed_detections, detection_map = index_detections(detections)

    # Check if indexed_detections has object_ids
    assert "0" in indexed_detections
    assert "10" in indexed_detections
    assert len(indexed_detections["0"]) == 2
    assert len(indexed_detections["10"]) == 1

    # Check if object_ids are assigned
    assert "object_id" in indexed_detections["0"][0]
    assert "object_id" in indexed_detections["0"][1]
    assert "object_id" in indexed_detections["10"][0]

    # Check if detection_map is correct
    obj_id_0 = indexed_detections["0"][0]["object_id"]
    obj_id_1 = indexed_detections["0"][1]["object_id"]
    obj_id_2 = indexed_detections["10"][0]["object_id"]

    assert obj_id_0 in detection_map
    assert obj_id_1 in detection_map
    assert obj_id_2 in detection_map

    assert detection_map[obj_id_0]["frame_key"] == "0"
    assert detection_map[obj_id_1]["frame_key"] == "0"
    assert detection_map[obj_id_2]["frame_key"] == "10"

    assert detection_map[obj_id_0]["detection"]["label"] == "person"
    assert detection_map[obj_id_1]["detection"]["label"] == "car"
    assert detection_map[obj_id_2]["detection"]["label"] == "person"


def test_extract_object_ids():
    """Test the extract_object_ids function."""
    # Test JSON array format
    ids = extract_object_ids('["obj_0", "obj_1", "obj_2"]')
    assert ids == ["obj_0", "obj_1", "obj_2"]

    # Test JSON array with objects
    ids = extract_object_ids('[{"object_id": "obj_0"}, {"object_id": "obj_1"}]')
    assert ids == ["obj_0", "obj_1"]

    # Test regex pattern
    ids = extract_object_ids("Found objects obj_0 and obj_1 in the image")
    assert set(ids) == set(["obj_0", "obj_1"])

    # Test bracketed IDs
    ids = extract_object_ids("Objects [obj_0] and [obj_1] are present")
    assert set(ids) == set(["obj_0", "obj_1"])

    # Test line-by-line format
    ids = extract_object_ids("obj_0\nobj_1\nobj_2")
    assert set(ids) == set(["obj_0", "obj_1", "obj_2"])

    # Test empty or invalid input
    ids = extract_object_ids("")
    assert ids == []

    ids = extract_object_ids("No object IDs here")
    assert ids == []


def test_get_objects_by_ids():
    """Test the get_objects_by_ids function."""
    # Create a detection map
    detection_map = {
        "obj_0": {
            "frame_key": "0",
            "detection": {"label": "person", "confidence": 0.9},
        },
        "obj_1": {"frame_key": "0", "detection": {"label": "car", "confidence": 0.8}},
        "obj_2": {
            "frame_key": "10",
            "detection": {"label": "person", "confidence": 0.85},
        },
    }

    # Get objects by IDs
    objects = get_objects_by_ids(["obj_0", "obj_2"], detection_map)

    # Check results
    assert len(objects) == 2

    assert objects[0]["frame_key"] == "0"
    assert objects[0]["detection"]["label"] == "person"
    assert objects[0]["detection"]["confidence"] == 0.9

    assert objects[1]["frame_key"] == "10"
    assert objects[1]["detection"]["label"] == "person"
    assert objects[1]["detection"]["confidence"] == 0.85

    # Test with non-existent IDs
    objects = get_objects_by_ids(["obj_0", "nonexistent"], detection_map)
    assert len(objects) == 1
    assert objects[0]["detection"]["label"] == "person"


def test_parse_explanation_response():
    """Test the parse_explanation_response function."""
    # Create a detection map
    detection_map = {
        "obj_0": {
            "frame_key": "0",
            "detection": {"label": "person", "confidence": 0.9},
        },
        "obj_1": {"frame_key": "0", "detection": {"label": "car", "confidence": 0.8}},
    }

    # Test with EXPLANATION and HIGHLIGHT_OBJECTS sections
    response = """
    EXPLANATION:
    I detected a person and a car in the image.

    HIGHLIGHT_OBJECTS:
    ["obj_0", "obj_1"]
    """

    explanation, highlighted = parse_explanation_response(response, detection_map)

    assert "I detected a person and a car in the image." in explanation
    assert "HIGHLIGHT_OBJECTS" not in explanation
    assert len(highlighted) == 2
    assert highlighted[0]["detection"]["label"] == "person"
    assert highlighted[1]["detection"]["label"] == "car"

    # Test with only EXPLANATION section
    response = "EXPLANATION: I detected a person and a car in the image."

    explanation, highlighted = parse_explanation_response(response, detection_map)

    assert "I detected a person and a car in the image." in explanation
    assert len(highlighted) == 0

    # Test without explicit sections
    response = "I detected a person and a car in the image."

    explanation, highlighted = parse_explanation_response(response, detection_map)

    assert explanation == "I detected a person and a car in the image."
    assert len(highlighted) == 0
