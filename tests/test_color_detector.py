"""
Comprehensive test suite for the ColorDetector class.
Tests all color detection methods with various inputs.
"""

import cv2
import numpy as np
import pytest

from langvio.vision.color_detection import ColorDetector

# Fixtures for different types of test images


@pytest.fixture
def solid_red_image():
    """Create a solid red image."""
    return np.ones((100, 100, 3), dtype=np.uint8) * np.array(
        [0, 0, 255], dtype=np.uint8
    )  # BGR


@pytest.fixture
def solid_green_image():
    """Create a solid green image."""
    return np.ones((100, 100, 3), dtype=np.uint8) * np.array(
        [0, 255, 0], dtype=np.uint8
    )  # BGR


@pytest.fixture
def solid_blue_image():
    """Create a solid blue image."""
    return np.ones((100, 100, 3), dtype=np.uint8) * np.array(
        [255, 0, 0], dtype=np.uint8
    )  # BGR


@pytest.fixture
def solid_black_image():
    """Create a solid black image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)  # BGR


@pytest.fixture
def solid_white_image():
    """Create a solid white image."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 255  # BGR


@pytest.fixture
def solid_gray_image():
    """Create a solid gray image."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 128  # BGR


@pytest.fixture
def solid_yellow_image():
    """Create a solid yellow image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 0  # B channel
    img[:, :, 1] = 255  # G channel
    img[:, :, 2] = 255  # R channel
    return img  # BGR


@pytest.fixture
def solid_cyan_image():
    """Create a solid cyan image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # B channel
    img[:, :, 1] = 255  # G channel
    img[:, :, 2] = 0  # R channel
    return img  # BGR


@pytest.fixture
def solid_magenta_image():
    """Create a solid magenta image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # B channel
    img[:, :, 1] = 0  # G channel
    img[:, :, 2] = 255  # R channel
    return img  # BGR


@pytest.fixture
def solid_orange_image():
    """Create a solid orange image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 0  # B channel
    img[:, :, 1] = 128  # G channel
    img[:, :, 2] = 255  # R channel
    return img  # BGR


@pytest.fixture
def split_red_blue_image():
    """Create a split red/blue image (half red, half blue)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :, 0] = 255  # Blue in top half (B channel)
    img[50:, :, 2] = 255  # Red in bottom half (R channel)
    return img  # BGR


@pytest.fixture
def split_red_green_blue_image():
    """Create an RGB split image (one-third red, one-third green, one-third blue)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:33, :, 2] = 255  # Red in top third (R channel)
    img[33:66, :, 1] = 255  # Green in middle third (G channel)
    img[66:, :, 0] = 255  # Blue in bottom third (B channel)
    return img  # BGR


@pytest.fixture
def gradient_image():
    """Create a red to blue gradient image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        blue_val = int(255 * (i / 100))
        red_val = int(255 * (1 - i / 100))
        img[i, :, 0] = blue_val  # B channel
        img[i, :, 2] = red_val  # R channel
    return img  # BGR


@pytest.fixture
def checkerboard_image():
    """Create a black and white checkerboard pattern."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(0, 100, 20):
        for j in range(0, 100, 20):
            if (i // 20 + j // 20) % 2 == 0:
                img[i : i + 20, j : j + 20] = 255
    return img  # BGR


@pytest.fixture
def rainbow_image():
    """Create a rainbow gradient image with all test colors."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Make sure to include a proper representation of magenta (B and R channels high)
    for i in range(100):
        # Rainbow spectrum including magenta
        if i >= 80:  # Add a magenta section
            img[i, :] = [255, 0, 255]  # BGR for magenta
        else:
            # Original rainbow code
            hue = int(180 * i / 100)
            hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
            img[i, :] = bgr_color
    return img  # BGR


@pytest.fixture
def noisy_red_image():
    """Create a predominantly red image with noise."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 2] = 200  # Predominantly red (R channel)
    # Add random noise
    noise = np.random.randint(0, 30, (100, 100, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    return img  # BGR


@pytest.fixture
def invalid_image():
    """Create an "invalid" image (0x0)."""
    return np.zeros((0, 0, 3), dtype=np.uint8)  # BGR


@pytest.fixture
def single_channel_image():
    """Create a single-channel (grayscale) image."""
    return np.ones((100, 100), dtype=np.uint8) * 128  # Grayscale


# Test cases for each method of the ColorDetector class


class TestColorDetector:
    """Tests for the ColorDetector class."""

    def test_detect_color_solid_basic_colors(
        self, solid_red_image, solid_green_image, solid_blue_image
    ):
        """Test detect_color with solid basic color images."""
        # Test red image
        color = ColorDetector.detect_color(solid_red_image)
        assert "red" in color.lower(), f"Expected 'red' in '{color}'"

        # Test green image
        color = ColorDetector.detect_color(solid_green_image)
        assert "green" in color.lower(), f"Expected 'green' in '{color}'"

        # Test blue image
        color = ColorDetector.detect_color(solid_blue_image)
        assert "blue" in color.lower(), f"Expected 'blue' in '{color}'"

    def test_detect_color_solid_grayscale(
        self, solid_black_image, solid_white_image, solid_gray_image
    ):
        """Test detect_color with solid grayscale images."""
        # Test black image
        color = ColorDetector.detect_color(solid_black_image)
        assert "black" in color.lower(), f"Expected 'black' in '{color}'"

        # Test white image
        color = ColorDetector.detect_color(solid_white_image)
        assert "white" in color.lower(), f"Expected 'white' in '{color}'"

        # Test gray image
        color = ColorDetector.detect_color(solid_gray_image)
        assert "gray" in color.lower(), f"Expected 'gray' in '{color}'"

    def test_detect_color_solid_secondary_colors(
        self,
        solid_yellow_image,
        solid_cyan_image,
        solid_magenta_image,
        solid_orange_image,
    ):
        """Test detect_color with solid secondary color images."""
        # Test yellow image
        color = ColorDetector.detect_color(solid_yellow_image)
        assert "yellow" in color.lower(), f"Expected 'yellow' in '{color}'"

        # Test cyan image
        color = ColorDetector.detect_color(solid_cyan_image)
        assert any(
            c in color.lower() for c in ["cyan", "teal", "turquoise", "blue"]
        ), f"Expected cyan-like color in '{color}'"

        # Test magenta image
        color = ColorDetector.detect_color(solid_magenta_image)
        assert any(
            c in color.lower() for c in ["magenta", "purple", "violet", "pink"]
        ), f"Expected magenta-like color in '{color}'"

        # Test orange image
        color = ColorDetector.detect_color(solid_orange_image)
        assert (
            "orange" in color.lower() or "red" in color.lower()
        ), f"Expected 'orange' or 'red' in '{color}'"

    def test_detect_color_with_return_all(self, solid_red_image, split_red_blue_image):
        """Test detect_color with return_all=True."""
        # Test solid color
        color_percentages = ColorDetector.detect_color(solid_red_image, return_all=True)

        # Should return a dictionary
        assert isinstance(color_percentages, dict)

        # Should have at least one color
        assert len(color_percentages) >= 1

        # Red should be dominant
        dominant_color = max(color_percentages.items(), key=lambda x: x[1])[0]
        assert (
            "red" in dominant_color.lower()
        ), f"Expected 'red' in dominant color '{dominant_color}'"

        # Test split image
        color_percentages = ColorDetector.detect_color(
            split_red_blue_image, return_all=True
        )

        # Should detect multiple colors
        assert len(color_percentages) >= 2

        # Should have both red and blue
        has_red = any("red" in color.lower() for color in color_percentages.keys())
        has_blue = any("blue" in color.lower() for color in color_percentages.keys())
        assert (
            has_red and has_blue
        ), f"Expected both red and blue colors, got: {list(color_percentages.keys())}"

    def test_detect_color_threshold(
        self, split_red_blue_image, split_red_green_blue_image
    ):
        """Test detect_color with different thresholds."""
        # With high threshold, split image might be detected as multicolored
        color = ColorDetector.detect_color(split_red_blue_image, threshold=0.7)
        assert (
            color.lower() == "multicolored" or color.lower() == "unknown"
        ), f"Expected 'multicolored' or 'unknown', got '{color}'"

        # With low threshold, should detect a single dominant color
        color = ColorDetector.detect_color(split_red_blue_image, threshold=0.1)
        assert (
            color.lower() != "multicolored" and color.lower() != "unknown"
        ), f"Expected a color name, got '{color}'"

        # With RGB split, should definitely be multicolored with any reasonable threshold
        color = ColorDetector.detect_color(split_red_green_blue_image, threshold=0.5)
        assert (
            color.lower() == "multicolored" or color.lower() == "unknown"
        ), f"Expected 'multicolored' or 'unknown', got '{color}'"

    def test_detect_colors_layered(
        self, solid_red_image, split_red_blue_image, rainbow_image
    ):
        """Test detect_colors_layered function."""
        # Test with solid color
        colors = ColorDetector.detect_colors_layered(solid_red_image, max_colors=3)

        # Should return a list
        assert isinstance(colors, list)

        # Should have at least one color
        assert len(colors) >= 1

        # First color should be red
        assert (
            "red" in colors[0].lower()
        ), f"Expected 'red' in first color '{colors[0]}'"

        # Test with split image
        colors = ColorDetector.detect_colors_layered(split_red_blue_image, max_colors=3)

        # Should have at least two colors
        assert len(colors) >= 2

        # Should have both red and blue
        has_red = any("red" in color.lower() for color in colors)
        has_blue = any("blue" in color.lower() for color in colors)
        assert has_red and has_blue, f"Expected both red and blue colors, got: {colors}"

        # Test with rainbow image
        colors = ColorDetector.detect_colors_layered(rainbow_image, max_colors=5)

        # Should have multiple colors
        assert len(colors) >= 3, f"Expected at least 3 colors, got: {len(colors)}"

        # Test max_colors parameter
        colors_limited = ColorDetector.detect_colors_layered(
            rainbow_image, max_colors=2
        )
        assert (
            len(colors_limited) <= 2
        ), f"Expected at most 2 colors, got: {len(colors_limited)}"

    def test_get_color_profile(
        self, solid_red_image, split_red_blue_image, noisy_red_image
    ):
        """Test get_color_profile function."""
        # Test with solid color
        profile = ColorDetector.get_color_profile(solid_red_image)

        # Check profile structure
        assert "dominant_color" in profile
        assert "color_percentages" in profile
        assert "is_multicolored" in profile
        assert "brightness" in profile
        assert "saturation" in profile

        # Check values for red image
        assert "red" in profile["dominant_color"].lower()
        assert not profile["is_multicolored"]
        assert profile["brightness"] > 0
        assert profile["saturation"] > 0

        # Test with split image
        profile = ColorDetector.get_color_profile(split_red_blue_image)

        # Split image should have multiple colors
        assert len(profile["color_percentages"]) >= 2

        # May or may not be marked as multicolored depending on implementation
        if profile["is_multicolored"]:
            assert len(profile["color_percentages"]) >= 2

        # Test with noisy image
        profile = ColorDetector.get_color_profile(noisy_red_image)

        # Dominant color should still be red despite the noise
        assert "red" in profile["dominant_color"].lower()

    def test_get_color_name(self):
        """Test get_color_name function."""
        # Test basic colors (BGR format)
        red_color = ColorDetector.get_color_name((0, 0, 255))
        assert "red" in red_color.lower()

        green_color = ColorDetector.get_color_name((0, 255, 0))
        assert "green" in green_color.lower()

        blue_color = ColorDetector.get_color_name((255, 0, 0))
        assert "blue" in blue_color.lower()

        # Test grayscale
        black_color = ColorDetector.get_color_name((0, 0, 0))
        assert "black" in black_color.lower()

        white_color = ColorDetector.get_color_name((255, 255, 255))
        assert "white" in white_color.lower()

        gray_color = ColorDetector.get_color_name((128, 128, 128))
        assert "gray" in gray_color.lower()

        # Test secondary colors
        yellow_color = ColorDetector.get_color_name((0, 255, 255))
        assert "yellow" in yellow_color.lower() or "gold" in yellow_color.lower()

        cyan_color = ColorDetector.get_color_name((255, 255, 0))
        assert any(
            c in cyan_color.lower() for c in ["cyan", "teal", "turquoise", "blue"]
        )

        magenta_color = ColorDetector.get_color_name((255, 0, 255))
        assert any(
            c in magenta_color.lower() for c in ["magenta", "purple", "violet", "pink"]
        )

        orange_color = ColorDetector.get_color_name((0, 128, 255))
        assert "orange" in orange_color.lower() or "red" in orange_color.lower()

    def test_visualize_colors(
        self, solid_red_image, split_red_blue_image, rainbow_image
    ):
        """Test visualize_colors function."""
        # Test with solid color
        vis_image = ColorDetector.visualize_colors(solid_red_image)
        assert vis_image.shape[2] == 3, "Visualization should be a 3-channel image"
        assert (
            vis_image.shape[0] > 0 and vis_image.shape[1] > 0
        ), "Visualization should have positive dimensions"

        # Test with split image
        vis_image = ColorDetector.visualize_colors(split_red_blue_image)
        assert vis_image.shape[2] == 3, "Visualization should be a 3-channel image"
        assert (
            vis_image.shape[0] > 0 and vis_image.shape[1] > 0
        ), "Visualization should have positive dimensions"

        # Test with rainbow image
        vis_image = ColorDetector.visualize_colors(rainbow_image)
        assert vis_image.shape[2] == 3, "Visualization should be a 3-channel image"
        assert (
            vis_image.shape[0] > 0 and vis_image.shape[1] > 0
        ), "Visualization should have positive dimensions"
        # Rainbow should have many colors, so visualization should be taller
        assert (
            vis_image.shape[0] > 100
        ), "Rainbow visualization should be taller to show many colors"

    def test_edge_cases(self, invalid_image, single_channel_image, solid_black_image):
        """Test ColorDetector with edge cases."""
        # Test with invalid image (0x0)
        color = ColorDetector.detect_color(invalid_image)
        assert (
            color == "unknown"
        ), f"Expected 'unknown' for invalid image, got '{color}'"

        # Test with single-channel image (should handle or fail gracefully)
        try:
            color = ColorDetector.detect_color(single_channel_image)
            # If it doesn't raise an exception, it should return a valid result
            assert color in [
                "unknown",
                "gray",
                "black",
                "white",
            ], f"Expected grayscale color, got '{color}'"
        except cv2.error:
            # It's okay if it raises a cv2.error for single-channel images
            pass

        # Test get_color_profile with invalid image
        profile = ColorDetector.get_color_profile(invalid_image)
        assert profile["dominant_color"] == "unknown"
        assert profile["color_percentages"] == {}
        assert not profile["is_multicolored"]
        assert profile["brightness"] == 0
        assert profile["saturation"] == 0

        # Test visualize_colors with invalid image
        vis_image = ColorDetector.visualize_colors(invalid_image)
        assert (
            vis_image.shape[2] == 3
        ), "Even for invalid input, should return a valid visualization"

        # Test find_objects_by_color with invalid image
        mask = ColorDetector.find_objects_by_color(invalid_image, "red")
        assert isinstance(
            mask, np.ndarray
        ), "Should return a numpy array even for invalid input"
        assert mask.size == 0, "Mask for invalid image should be empty"

    def test_color_ranges(self):
        """Test the color range definitions in ColorDetector."""
        # Check that COLOR_RANGES is defined
        assert hasattr(ColorDetector, "COLOR_RANGES")
        assert len(ColorDetector.COLOR_RANGES) > 0

        # Check structure of color ranges
        for color_range in ColorDetector.COLOR_RANGES:
            assert (
                len(color_range) == 3
            ), "Each color range should have 3 elements (lower, upper, name)"
            lower, upper, name = color_range

            # Check bounds
            assert len(lower) == 3, "Lower bound should have 3 elements (H, S, V)"
            assert len(upper) == 3, "Upper bound should have 3 elements (H, S, V)"

            # Check H, S, V ranges
            assert 0 <= lower[0] <= 180, "H lower bound should be between 0 and 180"
            assert 0 <= upper[0] <= 180, "H upper bound should be between 0 and 180"
            assert 0 <= lower[1] <= 255, "S lower bound should be between 0 and 255"
            assert 0 <= upper[1] <= 255, "S upper bound should be between 0 and 255"
            assert 0 <= lower[2] <= 255, "V lower bound should be between 0 and 255"
            assert 0 <= upper[2] <= 255, "V upper bound should be between 0 and 255"

            # Check name
            assert isinstance(name, str), "Color name should be a string"
            assert len(name) > 0, "Color name should not be empty"
