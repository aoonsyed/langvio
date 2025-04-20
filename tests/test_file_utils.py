import os
import tempfile

import pytest

from langvio.utils.file_utils import (create_temp_copy, ensure_directory,
                                      get_file_extension,
                                      get_files_in_directory, is_image_file,
                                      is_video_file)


def test_ensure_directory():
    """Test ensure_directory function."""
    with tempfile.TemporaryDirectory() as tempdir:
        test_dir = os.path.join(tempdir, "test_dir")

        # Directory shouldn't exist yet
        assert not os.path.exists(test_dir)

        # Create the directory
        ensure_directory(test_dir)

        # Directory should exist now
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)

        # Should not raise error when directory already exists
        ensure_directory(test_dir)


def test_get_file_extension():
    """Test get_file_extension function."""
    # Test with various file paths
    assert get_file_extension("file.txt") == ".txt"
    assert get_file_extension("path/to/file.jpg") == ".jpg"
    assert get_file_extension("file.with.multiple.dots.png") == ".png"
    assert get_file_extension("file_without_extension") == ""
    assert (
        get_file_extension("/absolute/path/file.MP4") == ".mp4"
    )  # Should be lowercase
    assert get_file_extension("") == ""


def test_is_image_file():
    """Test is_image_file function."""
    # Test with image file extensions
    assert is_image_file("image.jpg") == True
    assert is_image_file("image.jpeg") == True
    assert is_image_file("image.png") == True
    assert is_image_file("image.bmp") == True
    assert is_image_file("image.tiff") == True
    assert is_image_file("image.webp") == True

    # Test with non-image file extensions
    assert is_image_file("video.mp4") == False
    assert is_image_file("document.pdf") == False
    assert is_image_file("file_without_extension") == False

    # Test case insensitivity
    assert is_image_file("image.JPG") == True
    assert is_image_file("image.PNG") == True


def test_is_video_file():
    """Test is_video_file function."""
    # Test with video file extensions
    assert is_video_file("video.mp4") == True
    assert is_video_file("video.avi") == True
    assert is_video_file("video.mov") == True
    assert is_video_file("video.mkv") == True
    assert is_video_file("video.webm") == True

    # Test with non-video file extensions
    assert is_video_file("image.jpg") == False
    assert is_video_file("document.pdf") == False
    assert is_video_file("file_without_extension") == False

    # Test case insensitivity
    assert is_video_file("video.MP4") == True
    assert is_video_file("video.AVI") == True


def test_create_temp_copy():
    """Test create_temp_copy function."""
    # Create a temporary file with content
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(b"test content")

    try:
        # Create a temporary copy
        temp_copy = create_temp_copy(temp_path, delete=False)

        try:
            # Check if copy exists
            assert os.path.exists(temp_copy)

            # Check if extension is preserved
            assert os.path.splitext(temp_copy)[1] == os.path.splitext(temp_path)[1]

            # Check if content is copied
            with open(temp_copy, "rb") as f:
                content = f.read()
                assert content == b"test content"
        finally:
            # Clean up the copy
            if os.path.exists(temp_copy):
                os.remove(temp_copy)
    finally:
        # Clean up the original
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_get_files_in_directory():
    """Test get_files_in_directory function."""
    with tempfile.TemporaryDirectory() as tempdir:
        # Create some files
        file_paths = [
            os.path.join(tempdir, "file1.txt"),
            os.path.join(tempdir, "file2.jpg"),
            os.path.join(tempdir, "file3.mp4"),
            os.path.join(tempdir, "subdir"),
        ]

        for path in file_paths[:3]:  # Create the files
            with open(path, "w") as f:
                f.write("test")

        os.mkdir(file_paths[3])  # Create the subdirectory

        # Get all files (no extension filter)
        files = get_files_in_directory(tempdir)
        assert len(files) == 3  # Should not include the subdirectory
        assert set(files) == set(file_paths[:3])

        # Filter by extension
        files = get_files_in_directory(tempdir, extensions=[".txt"])
        assert len(files) == 1
        assert files[0] == file_paths[0]

        # Filter by multiple extensions
        files = get_files_in_directory(tempdir, extensions=[".jpg", ".mp4"])
        assert len(files) == 2
        assert set(files) == set(file_paths[1:3])

        # Non-existent directory
        files = get_files_in_directory(os.path.join(tempdir, "nonexistent"))
        assert files == []
