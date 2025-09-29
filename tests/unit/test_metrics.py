#!/usr/bin/env python3
"""
Unit tests for evaluation metrics.

These tests verify the proper computation of evaluation metrics
for classification, regression, and text generation tasks.
"""

import logging
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
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

# Optional dependencies for advanced metric validation
try:
    import numpy as np
    from sklearn import metrics as sk_metrics
    HAS_SKLEARN = True
    logger.info("scikit-learn available for reference metric calculation")
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available; reference validation will be skipped")

from src.eval.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_text_generation_metrics,
)


class TestClassificationMetrics:
    """Test suite for classification metrics."""

    def test_binary_classification(self) -> None:
        """Test metrics for binary classification."""
        preds = [1, 0, 1, 1, 0, 1, 0, 0]
        targets = [1, 0, 1, 0, 0, 1, 1, 0]
        probabilities = [0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4, 0.1]
        
        metrics = compute_classification_metrics(preds, targets, probabilities)
        logger.debug(f"Binary classification metrics: {metrics}")
        
        # Check basic metrics
        assert "accuracy" in metrics
        assert math.isclose(metrics["accuracy"], 0.75)  # 6 correct out of 8
        
        # Check precision, recall, F1
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert "precision_micro" in metrics
        assert "recall_micro" in metrics
        assert "f1_micro" in metrics
        assert "f1_weighted" in metrics
        
        # Check AUC if probabilities provided
        assert "roc_auc" in metrics
        
        # Validate against sklearn if available
        if HAS_SKLEARN:
            sk_accuracy = sk_metrics.accuracy_score(targets, preds)
            assert math.isclose(metrics["accuracy"], sk_accuracy)
            
            sk_f1_macro = sk_metrics.f1_score(targets, preds, average="macro")
            assert math.isclose(metrics["f1_macro"], sk_f1_macro, abs_tol=1e-5)

    def test_multiclass_classification(self) -> None:
        """Test metrics for multiclass classification."""
        # Predictions and targets with 6 correct out of 9
        preds = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        targets = [0, 1, 2, 0, 1, 2, 1, 1, 0]  # Changed to match 6/9 accuracy
        
        # One-hot encoded probabilities for multiclass
        probs = [
            [0.8, 0.1, 0.1],  # Pred 0 with 0.8 confidence
            [0.1, 0.7, 0.2],  # Pred 1 with 0.7 confidence
            [0.0, 0.3, 0.7],  # Pred 2 with 0.7 confidence
            [0.6, 0.3, 0.1],  # Pred 0 with 0.6 confidence
            [0.2, 0.6, 0.2],  # Pred 1 with 0.6 confidence
            [0.1, 0.2, 0.7],  # Pred 2 with 0.7 confidence
            [0.5, 0.3, 0.2],  # Pred 0 with 0.5 confidence
            [0.3, 0.5, 0.2],  # Pred 1 with 0.5 confidence
            [0.1, 0.1, 0.8],  # Pred 2 with 0.8 confidence
        ]
        
        metrics = compute_classification_metrics(preds, targets, probs)
        logger.debug(f"Multiclass classification metrics: {metrics}")
        
        # Count correct predictions
        correct = sum(1 for p, t in zip(preds, targets) if p == t)
        expected_accuracy = correct / len(preds)
        logger.debug(f"Expected accuracy: {expected_accuracy} ({correct}/{len(preds)})")
        
        # Check basic metrics
        assert "accuracy" in metrics
        assert math.isclose(metrics["accuracy"], expected_accuracy, abs_tol=1e-5)
        
        # Check multi-class metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert 0 <= metrics["precision_macro"] <= 1
        assert 0 <= metrics["recall_macro"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1

    def test_empty_predictions(self) -> None:
        """Test handling of empty predictions."""
        preds = []
        targets = []
        
        metrics = compute_classification_metrics(preds, targets)
        logger.debug(f"Empty classification metrics: {metrics}")
        
        # Should have default values
        assert "accuracy" in metrics
        assert metrics["accuracy"] == 0.0

    def test_mismatched_lengths(self) -> None:
        """Test handling of mismatched prediction and target lengths."""
        preds = [0, 1]
        targets = [0, 1, 0]
        
        with pytest.raises(ValueError, match="Predictions and targets must share the same length"):
            compute_classification_metrics(preds, targets)

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn required for this test")
    def test_probabilities_without_sklearn(self) -> None:
        """Test handling of probabilities when sklearn is not available."""
        # Run with probabilities
        preds = [0, 1, 1, 0]
        targets = [0, 1, 0, 0]
        probabilities = [0.2, 0.7, 0.6, 0.3]
        
        # Directly test the function with mocked sklearn import
        # We need to mock at the module level where it's imported
        with mock.patch.dict('sys.modules', {'sklearn': None}):
            # Re-import to get a fresh version without sklearn
            import importlib
            importlib.reload(sys.modules['src.eval.metrics'])
            from src.eval.metrics import compute_classification_metrics as compute_without_sklearn
            
            # This should use the fallback implementation
            metrics = compute_without_sklearn(preds, targets, probabilities)
            logger.debug(f"Metrics with sklearn unavailable: {metrics}")
            
            # Should still have basic metrics but not advanced ones like AUC
            assert "accuracy" in metrics
            assert "f1_macro" in metrics
            assert "roc_auc" not in metrics


class TestRegressionMetrics:
    """Test suite for regression metrics."""

    def test_basic_regression(self) -> None:
        """Test basic regression metrics."""
        preds = [2.0, 3.5, 4.0]
        targets = [2.0, 3.0, 5.0]
        
        metrics = compute_regression_metrics(preds, targets)
        logger.debug(f"Basic regression metrics: {metrics}")
        
        # Check metrics
        assert "mae" in metrics  # Mean Absolute Error
        assert "mse" in metrics  # Mean Squared Error
        assert "rmse" in metrics  # Root Mean Squared Error
        assert "r2" in metrics   # R-squared
        
        # MAE should be average of absolute differences
        expected_mae = sum(abs(p - t) for p, t in zip(preds, targets)) / len(preds)
        assert math.isclose(metrics["mae"], expected_mae)
        
        # RMSE should be square root of average squared differences
        expected_mse = sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(preds)
        expected_rmse = math.sqrt(expected_mse)
        assert math.isclose(metrics["rmse"], expected_rmse)
        
        # Validate with sklearn if available
        if HAS_SKLEARN:
            sk_mae = sk_metrics.mean_absolute_error(targets, preds)
            sk_mse = sk_metrics.mean_squared_error(targets, preds)
            sk_r2 = sk_metrics.r2_score(targets, preds)
            
            assert math.isclose(metrics["mae"], sk_mae)
            assert math.isclose(metrics["mse"], sk_mse)
            assert math.isclose(metrics["r2"], sk_r2)

    def test_perfect_predictions(self) -> None:
        """Test metrics with perfect predictions."""
        preds = [1.0, 2.0, 3.0, 4.0, 5.0]
        targets = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        metrics = compute_regression_metrics(preds, targets)
        logger.debug(f"Perfect regression metrics: {metrics}")
        
        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0

    def test_constant_predictions(self) -> None:
        """Test metrics with constant predictions."""
        preds = [3.0, 3.0, 3.0, 3.0]
        targets = [1.0, 2.0, 3.0, 4.0]
        
        metrics = compute_regression_metrics(preds, targets)
        logger.debug(f"Constant regression metrics: {metrics}")
        
        assert metrics["mae"] == 1.0  # Average absolute error is 1.0
        assert metrics["r2"] <= 0.0   # R2 should be negative or zero for constant predictions

    def test_empty_predictions_regression(self) -> None:
        """Test handling of empty predictions."""
        preds = []
        targets = []
        
        metrics = compute_regression_metrics(preds, targets)
        logger.debug(f"Empty regression metrics: {metrics}")
        
        # Should have default values
        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 0.0

    def test_mismatched_lengths_regression(self) -> None:
        """Test handling of mismatched prediction and target lengths."""
        preds = [1.0, 2.0]
        targets = [1.0, 2.0, 3.0]
        
        with pytest.raises(ValueError, match="Predictions and targets must share the same length"):
            compute_regression_metrics(preds, targets)


class TestTextGenerationMetrics:
    """Test suite for text generation metrics."""

    def test_basic_text_metrics(self) -> None:
        """Test basic text generation metrics."""
        preds = [
            "The cat sat on the mat.",
            "I like to eat pizza.",
            "Machine learning is fun."
        ]
        
        refs = [
            ["The cat sat on the mat."],  # Exact match
            ["I enjoy eating pizza."],     # Similar but not exact
            ["Natural language processing is interesting."]  # Different
        ]
        
        latencies = [100.0, 150.0, 200.0]  # ms
        
        # Mock nltk.word_tokenize to avoid NLTK data dependencies
        with mock.patch('nltk.word_tokenize', side_effect=lambda text: text.lower().split()):
            metrics = compute_text_generation_metrics(preds, refs, latencies_ms=latencies)
            logger.debug(f"Text generation metrics: {metrics}")
        
        # Check metrics
        assert "exact_match" in metrics
        assert metrics["exact_match"] == 1/3  # One exact match out of three
        
        # Latency metrics
        assert "latency_ms_avg" in metrics
        assert math.isclose(metrics["latency_ms_avg"], 150.0)
        assert "latency_ms_p50" in metrics
        assert "latency_ms_p90" in metrics
        assert "latency_ms_p99" in metrics
        
        # BLEU score
        assert "bleu" in metrics
        assert 0 <= metrics["bleu"] <= 1

    def test_multiple_references(self) -> None:
        """Test metrics with multiple references per prediction."""
        preds = [
            "The cat sat on the mat.",
            "The weather is nice today."
        ]
        
        refs = [
            ["A cat is on the mat.", "The cat sat on the mat.", "There is a cat on the mat."],  # Multiple refs
            ["It's a beautiful day.", "The weather is wonderful."]  # Multiple refs, no exact match
        ]
        
        # Mock nltk.word_tokenize to avoid NLTK data dependencies
        with mock.patch('nltk.word_tokenize', side_effect=lambda text: text.lower().split()):
            metrics = compute_text_generation_metrics(preds, refs)
            logger.debug(f"Multiple references metrics: {metrics}")
        
        assert metrics["exact_match"] == 0.5  # One exact match out of two
        assert "bleu" in metrics

    def test_empty_text_predictions(self) -> None:
        """Test handling of empty text predictions."""
        preds = []
        refs = []
        
        metrics = compute_text_generation_metrics(preds, refs)
        logger.debug(f"Empty text metrics: {metrics}")
        
        # Should have default values
        assert metrics["exact_match"] == 0.0
        assert metrics["bleu"] == 0.0

    def test_mismatched_lengths_text(self) -> None:
        """Test handling of mismatched prediction and reference lengths."""
        preds = ["Text one", "Text two"]
        refs = [["Reference one"]]
        
        with pytest.raises(ValueError, match="Predictions and references must share the same length"):
            compute_text_generation_metrics(preds, refs)

    def test_invalid_references_format(self) -> None:
        """Test handling of invalid references format."""
        preds = ["Text one", "Text two"]
        # References should be a list of lists, not a list of strings
        refs = ["Reference one", "Reference two"]
        
        with pytest.raises(ValueError, match="Each reference must be a list of strings"):
            compute_text_generation_metrics(preds, refs)


if __name__ == "__main__":
    # Run the tests directly if file is executed
    pytest.main(["-v", __file__])