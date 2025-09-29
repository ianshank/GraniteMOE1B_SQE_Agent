"""
Unit tests for path resolution in the constants module.
These tests verify that paths are correctly resolved regardless of the current working directory.
"""

import os
import unittest
from pathlib import Path
import tempfile
import shutil
import sys

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.constants import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_LOGS_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_REQUIREMENTS_DIR,
    DEFAULT_TRAINING_DIR,
    DEFAULT_USER_STORIES_DIR,
    _CURRENT_FILE,
    _MODULE_DIR,
    _SRC_DIR,
    _GRANITE_DIR,
    _PROJECT_ROOT
)


class TestPathResolution(unittest.TestCase):
    """Test case for path resolution in the constants module."""

    def setUp(self):
        """Set up the test environment."""
        # Save the original working directory
        self.original_dir = os.getcwd()
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after the test."""
        # Restore the original working directory
        os.chdir(self.original_dir)
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_path_hierarchy(self):
        """Test that the path hierarchy is correctly established."""
        # Verify the path hierarchy
        self.assertEqual(_MODULE_DIR, _CURRENT_FILE.parent)
        self.assertEqual(_SRC_DIR, _MODULE_DIR.parent)
        self.assertEqual(_GRANITE_DIR, _SRC_DIR.parent)
        self.assertEqual(_PROJECT_ROOT, _GRANITE_DIR.parent)

    def test_absolute_paths(self):
        """Test that all paths are absolute."""
        paths = [
            DEFAULT_OUTPUT_DIR,
            DEFAULT_MODELS_DIR,
            DEFAULT_CACHE_DIR,
            DEFAULT_LOGS_DIR,
            DEFAULT_DATA_DIR,
            DEFAULT_REQUIREMENTS_DIR,
            DEFAULT_TRAINING_DIR,
            DEFAULT_USER_STORIES_DIR
        ]
        
        for path in paths:
            self.assertTrue(os.path.isabs(path), f"Path {path} is not absolute")

    def test_path_resolution_with_changed_directory(self):
        """Test that paths are correctly resolved even when the working directory changes."""
        # Change the working directory
        os.chdir(self.temp_dir)
        
        # Verify that the output directory is still correctly resolved
        self.assertEqual(
            DEFAULT_OUTPUT_DIR,
            str(_PROJECT_ROOT / "output")
        )
        
        # Verify that the requirements directory is still correctly resolved
        self.assertEqual(
            DEFAULT_REQUIREMENTS_DIR,
            str(_GRANITE_DIR / "data/requirements")
        )

    def test_output_dir_points_to_project_root(self):
        """Test that the output directory points to the project root."""
        expected_output_dir = str(Path(_PROJECT_ROOT) / "output")
        self.assertEqual(DEFAULT_OUTPUT_DIR, expected_output_dir)


if __name__ == "__main__":
    unittest.main()
