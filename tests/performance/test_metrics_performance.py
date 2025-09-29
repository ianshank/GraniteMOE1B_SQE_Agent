#!/usr/bin/env python3
"""
Performance tests for evaluation metrics.

These tests verify the performance of the metrics computation
with large datasets to ensure they scale well.
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple
from unittest import mock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional dependencies for generating test data
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("NumPy not available; some performance tests will be skipped")

from src.eval.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_text_generation_metrics,
)


def generate_classification_data(size: int) -> Tuple[List[int], List[int], List[float]]:
    """Generate synthetic classification data."""
    if not HAS_NUMPY:
        # Fallback without NumPy
        predictions = [i % 2 for i in range(size)]
        targets = [(i + i // 3) % 2 for i in range(size)]
        probabilities = [0.5 + 0.4 * (i % 2) for i in range(size)]
    else:
        # Use NumPy for more realistic data
        np.random.seed(42)
        predictions = np.random.randint(0, 2, size=size).tolist()
        targets = np.random.randint(0, 2, size=size).tolist()
        probabilities = np.random.rand(size).tolist()
    
    return predictions, targets, probabilities


def generate_regression_data(size: int) -> Tuple[List[float], List[float]]:
    """Generate synthetic regression data."""
    if not HAS_NUMPY:
        # Fallback without NumPy
        predictions = [float(i) for i in range(size)]
        targets = [float(i) + (i % 5 - 2) for i in range(size)]
    else:
        # Use NumPy for more realistic data
        np.random.seed(42)
        x = np.linspace(0, 10, size)
        predictions = (2 * x + 1).tolist()
        targets = (2 * x + 1 + np.random.normal(0, 1, size)).tolist()
    
    return predictions, targets


def generate_text_data(size: int) -> Tuple[List[str], List[List[str]]]:
    """Generate synthetic text data."""
    words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew"]
    
    predictions = []
    references = []
    
    for i in range(size):
        # Generate a random sentence
        if not HAS_NUMPY:
            # Simple deterministic generation without NumPy
            pred_len = 3 + (i % 5)
            pred_words = [words[(i + j) % len(words)] for j in range(pred_len)]
            prediction = " ".join(pred_words)
            
            # Generate 1-3 references
            ref_count = 1 + (i % 3)
            refs = []
            for r in range(ref_count):
                ref_len = 3 + ((i + r) % 5)
                ref_words = [words[((i + j + r) % len(words))] for j in range(ref_len)]
                refs.append(" ".join(ref_words))
        else:
            # More varied generation with NumPy
            np.random.seed(i)  # Different seed for each item
            pred_len = np.random.randint(3, 8)
            pred_words = np.random.choice(words, size=pred_len).tolist()
            prediction = " ".join(pred_words)
            
            # Generate 1-3 references
            ref_count = np.random.randint(1, 4)
            refs = []
            for r in range(ref_count):
                ref_len = np.random.randint(3, 8)
                ref_words = np.random.choice(words, size=ref_len).tolist()
                refs.append(" ".join(ref_words))
        
        predictions.append(prediction)
        references.append(refs)
    
    return predictions, references


class TestMetricsPerformance:
    """Performance tests for metrics computation."""
    
    @pytest.mark.parametrize("size", [1000, 10000, 100000])
    def test_classification_metrics_performance(self, size: int, benchmark) -> None:
        """Test performance of classification metrics with different dataset sizes."""
        predictions, targets, probabilities = generate_classification_data(size)
        
        # Benchmark the computation
        def compute_metrics():
            return compute_classification_metrics(predictions, targets, probabilities)
        
        result = benchmark(compute_metrics)
        
        # Log the results
        logger.info(f"Classification metrics for size {size}: {result}")
        
        # Basic validation
        assert "accuracy" in result
        assert 0 <= result["accuracy"] <= 1
    
    @pytest.mark.parametrize("size", [1000, 10000, 100000])
    def test_regression_metrics_performance(self, size: int, benchmark) -> None:
        """Test performance of regression metrics with different dataset sizes."""
        predictions, targets = generate_regression_data(size)
        
        # Benchmark the computation
        def compute_metrics():
            return compute_regression_metrics(predictions, targets)
        
        result = benchmark(compute_metrics)
        
        # Log the results
        logger.info(f"Regression metrics for size {size}: {result}")
        
        # Basic validation
        assert "mae" in result
        assert result["mae"] >= 0
    
    @pytest.mark.parametrize("size", [100, 1000])
    def test_text_metrics_performance(self, size: int, benchmark) -> None:
        """Test performance of text metrics with different dataset sizes."""
        predictions, references = generate_text_data(size)
        
        # Mock nltk.word_tokenize to avoid NLTK data dependencies
        with mock.patch('nltk.word_tokenize', side_effect=lambda text: text.lower().split()):
            # Benchmark the computation
            def compute_metrics():
                return compute_text_generation_metrics(predictions, references)
            
            result = benchmark(compute_metrics)
        
        # Log the results
        logger.info(f"Text metrics for size {size}: {result}")
        
        # Basic validation
        assert "exact_match" in result
        assert 0 <= result["exact_match"] <= 1


if __name__ == "__main__":
    # Run the tests directly if file is executed
    pytest.main(["-v", __file__])