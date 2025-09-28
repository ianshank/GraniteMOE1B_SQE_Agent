import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


@pytest.mark.regression
def test_import_main():
    """Test that main.py can be imported without errors."""
    try:
        # Import the main module
        import main
        assert hasattr(main, 'main')
        assert callable(main.main)
    except ImportError as e:
        pytest.fail(f"Failed to import main module: {e}")


@pytest.mark.regression
def test_main_function():
    """Test that the main function runs without errors."""
    try:
        import main
        # Should not raise an exception
        main.main()
    except Exception as e:
        pytest.fail(f"Main function failed: {e}")


def test_src_directory_structure():
    """Test that the src directory structure is correct."""
    src_path = Path(__file__).parent.parent / "src"
    
    expected_dirs = [
        "agents",
        "data", 
        "integration",
        "models",
        "utils"
    ]
    
    for dir_name in expected_dirs:
        dir_path = src_path / dir_name
        assert dir_path.exists(), f"Directory {dir_name} does not exist in src/"
        assert dir_path.is_dir(), f"{dir_name} is not a directory"


def test_config_files_exist():
    """Test that configuration files exist."""
    config_path = Path(__file__).parent.parent / "config"
    
    expected_configs = [
        "model_config.yaml",
        "training_config.yaml", 
        "integration_config.yaml"
    ]
    
    for config_file in expected_configs:
        file_path = config_path / config_file
        assert file_path.exists(), f"Config file {config_file} does not exist"


def test_requirements_file():
    """Test that requirements.txt exists and is not empty."""
    req_path = Path(__file__).parent.parent / "requirements.txt"
    
    assert req_path.exists(), "requirements.txt does not exist"
    assert req_path.stat().st_size > 0, "requirements.txt is empty"


if __name__ == "__main__":
    pytest.main([__file__])
