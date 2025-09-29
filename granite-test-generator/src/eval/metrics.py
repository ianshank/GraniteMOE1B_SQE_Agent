"""Evaluation metrics for classification, regression, and text generation tasks."""

from typing import Any, Dict, List, Optional, Sequence, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Try to import sklearn metrics, but allow for graceful degradation
try:
    from sklearn import metrics as sk_metrics
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available; using basic metrics")


def compute_classification_metrics(
    predictions: Sequence[int],
    targets: Sequence[int],
    probabilities: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Compute metrics for classification tasks.
    
    Args:
        predictions: Predicted class indices
        targets: Target class indices
        probabilities: Optional prediction probabilities for positive class
            (for binary classification) or highest probability class
    
    Returns:
        Dictionary of metrics including accuracy, precision, recall, F1 score
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions and targets must share the same length: {len(predictions)} vs {len(targets)}")
    
    if not predictions:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "precision_micro": 0.0,
            "recall_micro": 0.0,
            "f1_micro": 0.0,
            "f1_weighted": 0.0,
        }
    
    # Try to use scikit-learn for more sophisticated metrics
    if HAS_SKLEARN:
        try:
            # Basic metrics
            accuracy = sk_metrics.accuracy_score(targets, predictions)
            
            # Precision, recall, F1
            precision_macro = sk_metrics.precision_score(targets, predictions, average="macro", zero_division=0)
            recall_macro = sk_metrics.recall_score(targets, predictions, average="macro", zero_division=0)
            f1_macro = sk_metrics.f1_score(targets, predictions, average="macro", zero_division=0)
            
            precision_micro = sk_metrics.precision_score(targets, predictions, average="micro", zero_division=0)
            recall_micro = sk_metrics.recall_score(targets, predictions, average="micro", zero_division=0)
            f1_micro = sk_metrics.f1_score(targets, predictions, average="micro", zero_division=0)
            
            f1_weighted = sk_metrics.f1_score(targets, predictions, average="weighted", zero_division=0)
            
            metrics = {
                "accuracy": accuracy,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
                "precision_micro": precision_micro,
                "recall_micro": recall_micro,
                "f1_micro": f1_micro,
                "f1_weighted": f1_weighted,
            }
            
            # Add AUC if probabilities are provided
            if probabilities is not None:
                try:
                    # Check if binary classification
                    unique_classes = len(set(targets))
                    if unique_classes <= 2:
                        roc_auc = sk_metrics.roc_auc_score(targets, probabilities)
                        metrics["roc_auc"] = roc_auc
                except Exception as e:
                    logger.warning(f"Could not compute ROC AUC: {e}")
            
            return metrics
        except Exception as e:
            logger.warning(f"Error using scikit-learn metrics: {e}, falling back to basic metrics")
    
    # Compute basic metrics without scikit-learn
    correct = sum(p == t for p, t in zip(predictions, targets))
    accuracy = correct / len(predictions)
    
    # Very basic precision/recall calculation
    # This is a simplified version and doesn't handle all edge cases
    classes = sorted(set(targets).union(set(predictions)))
    
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    
    for cls in classes:
        true_positives = sum(1 for p, t in zip(predictions, targets) if p == cls and t == cls)
        false_positives = sum(1 for p, t in zip(predictions, targets) if p == cls and t != cls)
        false_negatives = sum(1 for p, t in zip(predictions, targets) if p != cls and t == cls)
        
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
    
    return {
        "accuracy": accuracy,
        "precision_macro": precision_sum / len(classes),
        "recall_macro": recall_sum / len(classes),
        "f1_macro": f1_sum / len(classes),
        "precision_micro": accuracy,  # Simplified
        "recall_micro": accuracy,     # Simplified
        "f1_micro": accuracy,         # Simplified
        "f1_weighted": accuracy,      # Simplified
    }


def compute_regression_metrics(
    predictions: Sequence[float],
    targets: Sequence[float],
) -> Dict[str, float]:
    """
    Compute metrics for regression tasks.
    
    Args:
        predictions: Predicted values
        targets: Target values
    
    Returns:
        Dictionary of metrics including MAE, MSE, RMSE, and R²
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions and targets must share the same length: {len(predictions)} vs {len(targets)}")
    
    if not predictions:
        return {
            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
        }
    
    import math
    
    # Mean Absolute Error
    mae = sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)
    
    # Mean Squared Error
    mse = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
    
    # Root Mean Squared Error
    rmse = math.sqrt(mse)
    
    # R-squared
    mean_target = sum(targets) / len(targets)
    ss_total = sum((t - mean_target) ** 2 for t in targets)
    ss_residual = sum((t - p) ** 2 for t, p in zip(targets, predictions))
    
    # Handle edge case where all targets are the same
    if ss_total < 1e-10:
        r2 = 1.0 if ss_residual < 1e-10 else 0.0
    else:
        r2 = 1.0 - (ss_residual / ss_total)
        # R² can be negative if the model is worse than predicting the mean
        r2 = max(-1.0, r2)
    
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def compute_text_generation_metrics(
    predictions: Sequence[str],
    references: Sequence[List[str]],
    latencies_ms: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Compute metrics for text generation tasks.
    
    Args:
        predictions: Generated text outputs
        references: Lists of reference texts (multiple references per prediction)
        latencies_ms: Optional list of generation latencies in milliseconds
    
    Returns:
        Dictionary of metrics including exact match, BLEU, and latency statistics
    """
    if len(predictions) != len(references):
        raise ValueError(f"Predictions and references must share the same length: {len(predictions)} vs {len(references)}")
    
    # Validate references format
    for i, refs in enumerate(references):
        if not isinstance(refs, (list, tuple)):
            raise ValueError(f"Each reference must be a list of strings, but reference at index {i} is {type(refs)}")
    
    if not predictions:
        return {
            "exact_match": 0.0,
            "bleu": 0.0,
            "latency_ms_avg": 0.0,
            "latency_ms_p50": 0.0,
            "latency_ms_p90": 0.0,
            "latency_ms_p99": 0.0,
        }
    
    # Exact match calculation
    exact_matches = 0
    for pred, refs in zip(predictions, references):
        if any(pred == ref for ref in refs):
            exact_matches += 1
    
    exact_match_rate = exact_matches / len(predictions)
    
    # Try to compute BLEU score if nltk is available
    bleu_score = 0.0
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        # Ensure tokenizers are available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK punkt tokenizer: {e}")
                # Fall back to simple tokenization
                nltk.word_tokenize = lambda text: text.lower().split()
        
        # Compute BLEU for each prediction
        smoothing = SmoothingFunction().method1
        bleu_scores = []
        
        for pred, refs in zip(predictions, references):
            try:
                pred_tokens = nltk.word_tokenize(pred.lower())
                refs_tokens = [nltk.word_tokenize(ref.lower()) for ref in refs]
                
                # Skip empty predictions or references
                if not pred_tokens or not any(refs_tokens):
                    continue
                    
                score = sentence_bleu(refs_tokens, pred_tokens, smoothing_function=smoothing)
                bleu_scores.append(score)
            except Exception as e:
                logger.warning(f"Error computing BLEU: {e}")
        
        if bleu_scores:
            bleu_score = sum(bleu_scores) / len(bleu_scores)
    
    except ImportError:
        logger.warning("nltk not available; BLEU score set to 0")
    except Exception as e:
        logger.warning(f"Error in BLEU calculation: {e}")
    
    metrics = {
        "exact_match": exact_match_rate,
        "bleu": bleu_score,
    }
    
    # Add latency metrics if provided
    if latencies_ms:
        if len(latencies_ms) != len(predictions):
            logger.warning(f"Latency count ({len(latencies_ms)}) doesn't match prediction count ({len(predictions)})")
        
        latencies = sorted(latencies_ms)
        avg_latency = sum(latencies) / len(latencies)
        
        # Percentile calculation
        def percentile(values, p):
            """Calculate the pth percentile of values."""
            if not values:
                return 0.0
                
            k = (len(values) - 1) * (p / 100.0)
            f = int(k)
            c = min(f + 1, len(values) - 1)
            if f == c:
                return float(values[int(k)])
            d0 = values[f] * (c - k)
            d1 = values[c] * (k - f)
            return float(d0 + d1)
        
        metrics.update({
            "latency_ms_avg": avg_latency,
            "latency_ms_p50": percentile(latencies, 50),
            "latency_ms_p90": percentile(latencies, 90),
            "latency_ms_p99": percentile(latencies, 99),
        })
    
    return metrics