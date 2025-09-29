import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_text_generation_metrics,
)


def test_classification_metrics():
    preds = [0, 1, 1, 0]
    targets = [0, 1, 0, 0]
    probabilities = [0.2, 0.8, 0.6, 0.4]

    metrics = compute_classification_metrics(preds, targets, probabilities)
    assert math.isclose(metrics["accuracy"], 0.75)
    assert "precision_macro" in metrics
    assert "f1_micro" in metrics


def test_regression_metrics():
    preds = [2.0, 3.5, 4.0]
    targets = [2.0, 3.0, 5.0]

    metrics = compute_regression_metrics(preds, targets)
    assert math.isclose(metrics["mae"], 0.5)
    assert metrics["r2"] <= 1.0


def test_text_generation_metrics():
    preds = ["hello world", "lorem ipsum"]
    refs = [["hello world"], ["dolor sit"]]
    latencies = [10.0, 12.0]

    metrics = compute_text_generation_metrics(preds, refs, latencies_ms=latencies)
    assert math.isclose(metrics["exact_match"], 0.5)
    assert "latency_ms_avg" in metrics
