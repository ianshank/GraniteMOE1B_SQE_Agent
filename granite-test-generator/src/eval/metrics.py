"""
Metrics for evaluating ML models.

This module provides functions for computing metrics for various ML tasks:
- Classification (accuracy, precision, recall, F1, etc.)
- Regression (MSE, MAE, R2, etc.)
- Text generation (BLEU, ROUGE, etc.)
"""

import logging
import math
from typing import Dict, List, Optional, Union, Any

import numpy as np

# Download nltk data for text metrics
try:
    import nltk
    nltk.download('punkt')
except ImportError:
    pass

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    predictions: List[int],
    targets: List[int],
    probabilities: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute metrics for classification tasks.

    Args:
        predictions: List of predicted class indices
        targets: List of target class indices
        probabilities: Optional list of prediction probabilities for positive class

    Returns:
        Dictionary of metrics
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")

    metrics = {}

    # Basic metrics
    correct = sum(p == t for p, t in zip(predictions, targets))
    metrics["accuracy"] = correct / len(predictions) if predictions else 0.0

    # Try to use sklearn for advanced metrics if available
    try:
        from sklearn import metrics as sk_metrics

        # Get unique classes
        classes = sorted(set(targets) | set(predictions))
        is_binary = len(classes) <= 2
        is_multiclass = len(classes) > 2

        # Precision, recall, F1
        metrics["precision_macro"] = sk_metrics.precision_score(
            targets, predictions, average="macro", zero_division=0
        )
        metrics["recall_macro"] = sk_metrics.recall_score(
            targets, predictions, average="macro", zero_division=0
        )
        metrics["f1_macro"] = sk_metrics.f1_score(
            targets, predictions, average="macro", zero_division=0
        )
        metrics["f1_micro"] = sk_metrics.f1_score(
            targets, predictions, average="micro", zero_division=0
        )

        # Confusion matrix
        cm = sk_metrics.confusion_matrix(targets, predictions)
        metrics["true_positives"] = cm[1, 1] if is_binary and cm.shape == (2, 2) else None
        metrics["false_positives"] = cm[0, 1] if is_binary and cm.shape == (2, 2) else None
        metrics["true_negatives"] = cm[0, 0] if is_binary and cm.shape == (2, 2) else None
        metrics["false_negatives"] = cm[1, 0] if is_binary and cm.shape == (2, 2) else None

        # ROC AUC if probabilities provided
        if probabilities and is_binary:
            metrics["roc_auc"] = sk_metrics.roc_auc_score(targets, probabilities)

        # Multiclass metrics
        if is_multiclass:
            metrics["accuracy_balanced"] = sk_metrics.balanced_accuracy_score(targets, predictions)

    except ImportError:
        logger.warning("sklearn not available, computing basic metrics only")
    except Exception as e:
        logger.warning(f"Error computing sklearn metrics: {e}")

    # Remove None values
    metrics = {k: v for k, v in metrics.items() if v is not None}

    return metrics


def compute_regression_metrics(predictions: List[float], targets: List[float]) -> Dict[str, float]:
    """
    Compute metrics for regression tasks.

    Args:
        predictions: List of predicted values
        targets: List of target values

    Returns:
        Dictionary of metrics
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")

    metrics = {}

    # Basic metrics
    errors = [p - t for p, t in zip(predictions, targets)]
    abs_errors = [abs(e) for e in errors]
    squared_errors = [e * e for e in errors]

    metrics["mse"] = sum(squared_errors) / len(squared_errors) if squared_errors else 0.0
    metrics["rmse"] = math.sqrt(metrics["mse"])
    metrics["mae"] = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0

    # Try to use sklearn for advanced metrics if available
    try:
        from sklearn import metrics as sk_metrics

        metrics["r2"] = sk_metrics.r2_score(targets, predictions)
        metrics["explained_variance"] = sk_metrics.explained_variance_score(targets, predictions)
    except ImportError:
        logger.warning("sklearn not available, computing basic metrics only")
        
        # Compute R2 manually
        if len(targets) > 1:
            mean_target = sum(targets) / len(targets)
            total_variance = sum((t - mean_target) ** 2 for t in targets)
            if total_variance > 0:
                metrics["r2"] = 1.0 - (sum(squared_errors) / total_variance)
            else:
                metrics["r2"] = 0.0
        else:
            metrics["r2"] = 0.0
    except Exception as e:
        logger.warning(f"Error computing sklearn metrics: {e}")

    return metrics


def compute_text_generation_metrics(
    predictions: List[str],
    references: List[List[str]],
    latencies_ms: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute metrics for text generation tasks.

    Args:
        predictions: List of generated texts
        references: List of lists of reference texts (multiple references per example)
        latencies_ms: Optional list of generation latencies in milliseconds

    Returns:
        Dictionary of metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    metrics = {}

    # Exact match
    exact_matches = sum(
        any(pred.strip() == ref.strip() for ref in refs) for pred, refs in zip(predictions, references)
    )
    metrics["exact_match"] = exact_matches / len(predictions) if predictions else 0.0

    # Latency metrics
    if latencies_ms:
        metrics["latency_ms_avg"] = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        metrics["latency_ms_p50"] = _percentile(latencies_ms, 50)
        metrics["latency_ms_p90"] = _percentile(latencies_ms, 90)
        metrics["latency_ms_p99"] = _percentile(latencies_ms, 99)

    # Try to use nltk for advanced metrics if available
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize

        # BLEU scores
        bleu_scores = []
        for pred, refs in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            refs_tokens = [word_tokenize(ref.lower()) for ref in refs]
            if pred_tokens and any(refs_tokens):
                bleu_scores.append(sentence_bleu(refs_tokens, pred_tokens))
            else:
                bleu_scores.append(0.0)

        metrics["bleu_avg"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    except ImportError:
        logger.warning("nltk not available, skipping BLEU metrics")
    except Exception as e:
        logger.warning(f"Error computing nltk metrics: {e}")

    # Try to use rouge for advanced metrics if available
    try:
        from rouge import Rouge

        rouge = Rouge()
        rouge_scores = []
        for pred, refs in zip(predictions, references):
            if not pred.strip() or not any(ref.strip() for ref in refs):
                continue
            # Use first reference for now
            ref = next((r for r in refs if r.strip()), "")
            if not ref:
                continue
            try:
                scores = rouge.get_scores(pred, ref)[0]
                rouge_scores.append(scores)
            except Exception:
                # Skip examples that cause rouge errors
                continue

        if rouge_scores:
            metrics["rouge1_f"] = sum(s["rouge-1"]["f"] for s in rouge_scores) / len(rouge_scores)
            metrics["rouge2_f"] = sum(s["rouge-2"]["f"] for s in rouge_scores) / len(rouge_scores)
            metrics["rougeL_f"] = sum(s["rouge-l"]["f"] for s in rouge_scores) / len(rouge_scores)
    except ImportError:
        logger.warning("rouge not available, skipping ROUGE metrics")
    except Exception as e:
        logger.warning(f"Error computing rouge metrics: {e}")

    return metrics


def _percentile(values: List[float], p: float) -> float:
    """Compute the p-th percentile of a list of values."""
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values_sorted[int(k)]
    d0 = values_sorted[int(f)] * (c - k)
    d1 = values_sorted[int(c)] * (k - f)
    return d0 + d1
