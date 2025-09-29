#!/usr/bin/env python3
"""
Unit tests for the experiment logger.

These tests verify the proper functioning of the ExperimentLogger class
with real file system interactions and handling of various data types.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Configure logging for debug output during tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    logger.warning("NumPy not available for testing")
    HAS_NUMPY = False

from src.config import TelemetryConfig
from src.telemetry import ExperimentLogger


class TestExperimentLogger:
    """Test suite for the ExperimentLogger class."""

    def test_file_system_artifacts(self, tmp_path: Path) -> None:
        """Test handling of file system artifacts."""
        # Create test files
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        test_file = artifact_dir / "test_file.txt"
        test_file.write_text("This is a test file")
        
        # Create JSON file
        json_file = artifact_dir / "metrics.json"
        json_data = {"accuracy": 0.92, "f1": 0.88}
        with open(json_file, "w") as f:
            json.dump(json_data, f)
        
        # Configure logger with TensorBoard enabled
        config = TelemetryConfig(
            enable_tensorboard=True,
            tb_log_dir=str(tmp_path / "runs"),
        )
        
        # Use real filesystem interactions
        with ExperimentLogger(config, {"model": {"type": "test"}}) as exp_logger:
            # Log artifacts
            exp_logger.log_artifact(test_file)
            exp_logger.log_artifact(json_file, name="results", type="metrics")
            
            # Log metrics and parameters
            exp_logger.log_metrics(1, accuracy=0.92, f1=0.88)
            exp_logger.log_params(learning_rate=0.01, batch_size=32)
            exp_logger.set_summary(final_accuracy=0.95)
            
        # Verify TensorBoard output directory was created
        tb_dir = tmp_path / "runs"
        assert tb_dir.exists()
        assert any(tb_dir.glob("*"))  # Should contain at least one file
        logger.debug(f"TensorBoard directory contents: {list(tb_dir.glob('*'))}")

    def test_metrics_with_different_types(self, tmp_path: Path) -> None:
        """Test logging metrics with different data types."""
        config = TelemetryConfig(
            enable_tensorboard=True,
            tb_log_dir=str(tmp_path / "runs"),
        )
        
        with ExperimentLogger(config, {"test": True}) as exp_logger:
            # Log various types of metrics
            metrics = {
                "float_value": 0.12345,
                "int_value": 42,
                "bool_value": True,  # Should be converted to 1.0
                "string_value": "invalid",  # Should be filtered out
                "none_value": None,  # Should be filtered out
                "infinity": float("inf"),  # Should be filtered out
            }
            
            if HAS_NUMPY:
                metrics.update({
                    "np_float": np.float32(0.5),
                    "np_int": np.int32(5),
                    "np_array": np.array([1, 2, 3]),  # Should be filtered out
                })
            
            exp_logger.log_metrics(1, **metrics)
            
            # Log parameter dictionary
            params = {
                "learning_rate": 0.001,
                "optimizer": "adam",
                "epochs": 10,
                "nested": {"key": "value"},
            }
            exp_logger.log_params(**params)
            
            # Test summary with mixed types
            summary = {
                "best_accuracy": 0.95,
                "best_epoch": 5,
                "model_type": "transformer",
            }
            exp_logger.set_summary(**summary)

    def test_run_name_generation(self) -> None:
        """Test automatic run name generation."""
        # Test with minimal config
        config = TelemetryConfig()
        exp_logger1 = ExperimentLogger(config, {"model": {"type": "test"}})
        # Explicitly start the run to initialize _run_name
        exp_logger1.start_run()
        name1 = exp_logger1._run_name
        assert name1.startswith("test-dataset-")
        assert len(name1) > 15  # Should include timestamp
        exp_logger1.finish()
        
        # Test with different config values
        config = TelemetryConfig()
        exp_logger2 = ExperimentLogger(
            config,
            {
                "model_name": "custom_model",
                "dataset": "custom_dataset",
            }
        )
        # Explicitly start the run
        exp_logger2.start_run()
        name2 = exp_logger2._run_name
        assert name2.startswith("custom_model-custom_dataset-")
        exp_logger2.finish()
        
        # Test with explicit run name
        exp_logger3 = ExperimentLogger(
            config,
            {"model": {"type": "test"}},
            run_name="explicit-name"
        )
        # Explicitly start the run
        exp_logger3.start_run()
        assert exp_logger3._run_name == "explicit-name"
        exp_logger3.finish()
        
        # Test with string model value
        exp_logger4 = ExperimentLogger(
            config,
            {"model": "string-model"}
        )
        # Explicitly start the run
        exp_logger4.start_run()
        name4 = exp_logger4._run_name
        assert name4.startswith("string-model-dataset-")
        exp_logger4.finish()

    def test_git_sha_detection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test git SHA detection."""
        # Mock subprocess to return a fixed SHA
        def mock_check_output(args, **kwargs):
            return b"abcdef1234567890"
        
        import subprocess
        monkeypatch.setattr(subprocess, "check_output", mock_check_output)
        
        config = TelemetryConfig()
        exp_logger = ExperimentLogger(config, {})
        exp_logger.start_run()
        
        assert exp_logger._git_sha == "abcdef1234567890"
        exp_logger.finish()
        
        # Test when git command fails
        def mock_check_output_fail(args, **kwargs):
            raise subprocess.SubprocessError("Command failed")
        
        monkeypatch.setattr(subprocess, "check_output", mock_check_output_fail)
        
        exp_logger = ExperimentLogger(config, {})
        exp_logger.start_run()
        
        assert exp_logger._git_sha is None
        exp_logger.finish()

    def test_number_validation(self) -> None:
        """Test validation of numeric values."""
        is_number = ExperimentLogger._is_number
        
        assert is_number(42) is True
        assert is_number(3.14) is True
        assert is_number("123") is True
        assert is_number("-0.5") is True
        
        assert is_number("hello") is False
        assert is_number(None) is False
        assert is_number([]) is False
        assert is_number({}) is False

    def test_artifact_with_nonexistent_path(self, caplog) -> None:
        """Test handling of non-existent artifact paths."""
        caplog.set_level(logging.WARNING)
        
        config = TelemetryConfig()
        with ExperimentLogger(config, {}) as exp_logger:
            exp_logger.log_artifact("/path/does/not/exist.txt")
        
        assert "does not exist" in caplog.text
        assert "skipping upload" in caplog.text

    def test_graceful_multiple_finish(self) -> None:
        """Test graceful handling of multiple finish() calls."""
        config = TelemetryConfig()
        exp_logger = ExperimentLogger(config, {})
        
        # First finish should work normally
        exp_logger.finish()
        assert exp_logger._closed is True
        
        # Second finish should be a no-op
        exp_logger.finish()
        assert exp_logger._closed is True

    def test_no_metrics(self) -> None:
        """Test logging empty metrics dictionaries."""
        config = TelemetryConfig()
        with ExperimentLogger(config, {}) as exp_logger:
            # Empty metrics should be handled gracefully
            exp_logger.log_metrics(1)
            exp_logger.log_params()
            exp_logger.set_summary()
            
            # Zero or negative step should be handled
            exp_logger.log_metrics(0, test=1.0)
            exp_logger.log_metrics(-1, test=2.0)

    def test_experiment_logger_with_real_directory_structure(self, tmp_path: Path) -> None:
        """Test ExperimentLogger with a real directory structure."""
        # Create directory structure
        output_dir = tmp_path / "experiment"
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        
        # Create a checkpoint file
        checkpoint_file = checkpoints_dir / "model.pt"
        checkpoint_file.write_text("dummy checkpoint")
        
        # Create an evaluation report
        eval_dir = output_dir / "eval"
        eval_dir.mkdir()
        eval_report = eval_dir / "eval_report.json"
        eval_data = {"accuracy": 0.95, "f1": 0.93}
        with open(eval_report, "w") as f:
            json.dump(eval_data, f)
        
        # Configure telemetry
        config = TelemetryConfig(
            enable_tensorboard=True,
            tb_log_dir=str(tmp_path / "tb_logs"),
        )
        
        # Create a complete experiment workflow
        # Use a dictionary for model instead of a string
        with ExperimentLogger(config, {"model": {"type": "test"}}) as exp_logger:
            # Log training metrics
            for step in range(1, 6):
                exp_logger.log_metrics(
                    step,
                    loss=1.0 - step * 0.15,
                    accuracy=0.5 + step * 0.08,
                )
                # Simulate some processing time
                time.sleep(0.01)
            
            # Log hyperparameters
            exp_logger.log_params(
                learning_rate=0.01,
                batch_size=32,
                optimizer="adam",
                epochs=5,
            )
            
            # Log artifacts
            exp_logger.log_artifact(checkpoint_file, name="model_checkpoint", type="model")
            exp_logger.log_artifact(eval_report, name="evaluation_report", type="eval")
            
            # Set summary metrics
            exp_logger.set_summary(
                final_loss=0.25,
                final_accuracy=0.92,
                training_time=1.23,
            )
        
        # Verify that TensorBoard logs were created
        tb_logs_dir = tmp_path / "tb_logs"
        assert tb_logs_dir.exists()
        assert any(tb_logs_dir.glob("*"))


if __name__ == "__main__":
    # Run the tests directly if file is executed
    pytest.main(["-v", __file__])