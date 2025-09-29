#!/usr/bin/env python3
"""
Unit tests for evaluation helpers.

These tests verify the proper functioning of the evaluation helpers
for model assessment and metric computation.
"""

import logging
import sys
import math
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Configure logging for test debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    HAS_TORCH = True
    logger.info("PyTorch available for testing")
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available; some tests will be skipped")

from src.config import TelemetryConfig
from src.telemetry import ExperimentLogger
from src.eval.evaluate import (
    evaluate,
    _split_batch,
    _forward,
    _flatten,
    _flatten_probabilities,
    _percentile,
    _is_number,
)


class TestEvaluationHelpers:
    """Test suite for evaluation helper functions."""

    def test_split_batch_tuple(self) -> None:
        """Test splitting a batch tuple."""
        batch = ([1, 2, 3], [4, 5, 6])
        inputs, targets = _split_batch(batch)
        assert inputs == [1, 2, 3]
        assert targets == [4, 5, 6]

    def test_split_batch_dict(self) -> None:
        """Test splitting a batch dictionary."""
        batch = {"inputs": [1, 2, 3], "labels": [4, 5, 6]}
        inputs, targets = _split_batch(batch)
        assert inputs == [1, 2, 3]
        assert targets == [4, 5, 6]

    def test_split_batch_dict_alternative_keys(self) -> None:
        """Test splitting a batch dictionary with alternative keys."""
        batch = {"features": [1, 2, 3], "targets": [4, 5, 6]}
        inputs, targets = _split_batch(batch)
        assert inputs == [1, 2, 3]
        assert targets == [4, 5, 6]

    def test_split_batch_error_missing_keys(self) -> None:
        """Test error when batch dictionary is missing required keys."""
        batch = {"something": [1, 2, 3]}
        with pytest.raises(ValueError, match="Batch mapping must contain 'inputs'/'labels' keys"):
            _split_batch(batch)

    def test_split_batch_error_wrong_type(self) -> None:
        """Test error when batch is not a tuple or dictionary."""
        batch = "not a batch"
        with pytest.raises(TypeError, match="Unsupported batch structure for evaluation"):
            _split_batch(batch)

    def test_forward_callable_model(self) -> None:
        """Test forwarding with a callable model."""
        model = lambda x: x * 2
        inputs = 5
        result = _forward(model, inputs, "cpu")
        assert result == 10

    def test_forward_model_with_eval_step(self) -> None:
        """Test forwarding with a model that has eval_step method."""
        class ModelWithEvalStep:
            def eval_step(self, inputs, device=None):
                return inputs * 3, inputs * 0.5
        
        model = ModelWithEvalStep()
        inputs = 4
        result = _forward(model, inputs, "cpu")
        assert result == (12, 2.0)

    def test_forward_error_invalid_model(self) -> None:
        """Test error when model is not callable and has no eval_step."""
        model = "not a model"
        inputs = 5
        with pytest.raises(TypeError, match="Model must implement 'eval_step' or be callable"):
            _forward(model, inputs, "cpu")

    def test_flatten_simple_values(self) -> None:
        """Test flattening simple values."""
        assert _flatten(5) == [5]
        assert _flatten("test") == ["test"]
        assert _flatten(None) == [None]

    def test_flatten_lists(self) -> None:
        """Test flattening nested lists."""
        assert _flatten([1, 2, 3]) == [1, 2, 3]
        assert _flatten([1, [2, 3], 4]) == [1, 2, 3, 4]
        assert _flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]
        assert _flatten([]) == []

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_flatten_torch_tensor(self) -> None:
        """Test flattening PyTorch tensors."""
        tensor = torch.tensor([1, 2, 3])
        assert _flatten(tensor) == [1, 2, 3]
        
        tensor_2d = torch.tensor([[1, 2], [3, 4]])
        assert _flatten(tensor_2d) == [1, 2, 3, 4]

    def test_flatten_probabilities(self) -> None:
        """Test flattening probability arrays."""
        # Simple array matching expected length
        probs = [0.1, 0.2, 0.3]
        assert _flatten_probabilities(probs, 3) == [0.1, 0.2, 0.3]
        
        # 2D array with expected length
        probs_2d = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]
        assert _flatten_probabilities(probs_2d, 3) == [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]
        
        # Binary case with 2 values
        binary = [0.1, 0.9]
        assert _flatten_probabilities(binary, 1) == [0.9]  # Take second value (positive class)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_flatten_probabilities_torch(self) -> None:
        """Test flattening probability tensors."""
        # Tensor with exact match length
        tensor = torch.tensor([0.1, 0.2, 0.3])
        flattened = _flatten_probabilities(tensor, 3)
        # Use isclose for floating point comparison
        assert len(flattened) == 3
        assert math.isclose(flattened[0], 0.1, abs_tol=1e-5)
        assert math.isclose(flattened[1], 0.2, abs_tol=1e-5)
        assert math.isclose(flattened[2], 0.3, abs_tol=1e-5)
        
        # 2D tensor
        tensor_2d = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
        flattened_2d = _flatten_probabilities(tensor_2d, 3)
        assert len(flattened_2d) == 6  # Flattened to 1D

    def test_percentile(self) -> None:
        """Test percentile calculation."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Test median (50th percentile)
        assert _percentile(values, 50) == 5.5
        
        # Test quartiles
        assert _percentile(values, 25) == 3.25
        assert _percentile(values, 75) == 7.75
        
        # Test edge cases
        assert _percentile(values, 0) == 1.0
        assert _percentile(values, 100) == 10.0
        
        # Test empty list
        assert _percentile([], 50) == 0.0

    def test_is_number(self) -> None:
        """Test number validation."""
        assert _is_number(42) is True
        assert _is_number(3.14) is True
        assert _is_number("123") is True
        assert _is_number("-0.5") is True
        
        assert _is_number("hello") is False
        assert _is_number(None) is False
        assert _is_number([]) is False
        assert _is_number({}) is False

    def test_evaluate_classification(self, tmp_path: Path) -> None:
        """Test evaluate function with classification task."""
        # Simple model that always predicts class 1
        model = lambda x: 1
        
        # Simple dataloader with 5 samples
        class SimpleDataLoader:
            def __init__(self):
                self.data = [
                    ([0], 1),  # Correct
                    ([0], 0),  # Incorrect
                    ([0], 1),  # Correct
                    ([0], 1),  # Correct
                    ([0], 0),  # Incorrect
                ]
            
            def __iter__(self):
                return iter(self.data)
        
        dataloader = SimpleDataLoader()
        output_dir = tmp_path / "eval_results"
        
        # Create experiment logger
        config = TelemetryConfig()
        exp_logger = ExperimentLogger(config, {"model": "test_model"})
        exp_logger.start_run()  # Ensure run is started
        
        # Run evaluation
        metrics = evaluate(
            model,
            dataloader,
            task_type="classification",
            experiment_logger=exp_logger,
            output_dir=output_dir,
            epoch=1,
        )
        
        # Check metrics
        logger.debug(f"Classification metrics: {metrics}")
        assert "accuracy" in metrics
        assert math.isclose(metrics["accuracy"], 0.6)  # 3 correct out of 5
        
        # Check that report was created
        report_path = output_dir / "eval_report.json"
        assert report_path.exists()
        
        # Clean up
        exp_logger.finish()

    def test_evaluate_regression(self, tmp_path: Path) -> None:
        """Test evaluate function with regression task."""
        # Simple model that always predicts y = 2x
        model = lambda x: float(x[0]) * 2
        
        # Simple dataloader with regression samples
        class RegressionDataLoader:
            def __init__(self):
                self.data = [
                    ([1], 2.0),    # Correct
                    ([2], 3.0),    # Off by 1
                    ([3], 6.0),    # Correct
                    ([4], 7.0),    # Off by 1
                    ([5], 10.0),   # Correct
                ]
            
            def __iter__(self):
                return iter(self.data)
        
        dataloader = RegressionDataLoader()
        output_dir = tmp_path / "regression_results"
        
        # Create experiment logger
        config = TelemetryConfig()
        exp_logger = ExperimentLogger(config, {"model": "regression_model"})
        exp_logger.start_run()  # Ensure run is started
        
        # Run evaluation
        metrics = evaluate(
            model,
            dataloader,
            task_type="regression",
            experiment_logger=exp_logger,
            output_dir=output_dir,
            epoch=1,
        )
        
        # Check metrics
        logger.debug(f"Regression metrics: {metrics}")
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        
        # Check that report was created
        report_path = output_dir / "eval_report.json"
        assert report_path.exists()
        
        # Clean up
        exp_logger.finish()

    def test_evaluate_text(self, tmp_path: Path) -> None:
        """Test evaluate function with text generation task."""
        # Model that returns fixed responses
        responses = ["Hello world", "This is a test", "Machine learning"]
        iter_responses = iter(responses)
        model = lambda x: next(iter_responses)
        
        # Simple dataloader with text samples
        class TextDataLoader:
            def __init__(self):
                self.data = [
                    (["prompt1"], ["Hello world"]),          # Correct
                    (["prompt2"], ["Something different"]),  # Incorrect
                    (["prompt3"], ["Machine learning"]),     # Correct
                ]
            
            def __iter__(self):
                return iter(self.data)
        
        dataloader = TextDataLoader()
        output_dir = tmp_path / "text_results"
        
        # Create experiment logger
        config = TelemetryConfig()
        exp_logger = ExperimentLogger(config, {"model": "text_model"})
        exp_logger.start_run()  # Ensure run is started
        
        # Mock nltk.word_tokenize to avoid NLTK data dependencies
        with mock.patch('nltk.word_tokenize', side_effect=lambda text: text.lower().split()):
            # Run evaluation
            metrics = evaluate(
                model,
                dataloader,
                task_type="text",
                experiment_logger=exp_logger,
                output_dir=output_dir,
                epoch=1,
            )
        
        # Check metrics
        logger.debug(f"Text metrics: {metrics}")
        assert "exact_match" in metrics
        assert metrics["exact_match"] == 2/3  # 2 correct out of 3
        
        # Check that report was created
        report_path = output_dir / "eval_report.json"
        assert report_path.exists()
        
        # Clean up
        exp_logger.finish()

    def test_evaluate_error_invalid_task(self) -> None:
        """Test evaluate function with invalid task type."""
        model = lambda x: x
        dataloader = [(1, 1)]  # Simple dataloader with one sample
        
        # Use try-except to catch the error inside evaluate
        # The function now logs the error and returns an error dict
        metrics = evaluate(
            model,
            dataloader,
            task_type="invalid_task",
            output_dir=Path("dummy"),
        )
        
        # Check that error is returned
        assert "error" in metrics
        assert metrics["error"] == 1.0


if __name__ == "__main__":
    # Run the tests directly if file is executed
    pytest.main(["-v", __file__])