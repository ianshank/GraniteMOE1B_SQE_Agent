"""Evaluation metrics for classification, regression, and text generation tasks."""

from typing import Any, Dict, List, Optional, Sequence, Union
import logging
import math

# Configure logging
logger = logging.getLogger(__name__)

# Try to import sklearn metrics, but allow for graceful degradation
try:
    from sklearn import metrics as sk_metrics
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available; using basic metrics")

# Try to import numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy not available; using basic operations")

# Download nltk data for text metrics
try:
    import nltk
    nltk.download('punkt', quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    logger.warning("nltk not available; skipping advanced text metrics")


def compute_classification_metrics(
    predictions: Union[Sequence[int], List[int]],
    targets: Union[Sequence[int], List[int]],
    probabilities: Optional[Union[Sequence[float], List[float]]] = None,
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
            
            # Get unique classes for additional metrics
            classes = sorted(set(targets) | set(predictions))
            is_binary = len(classes) <= 2
            is_multiclass = len(classes) > 2

            # Confusion matrix
            cm = sk_metrics.confusion_matrix(targets, predictions)
            if is_binary and cm.shape == (2, 2):
                metrics["true_positives"] = float(cm[1, 1])
                metrics["false_positives"] = float(cm[0, 1])
                metrics["true_negatives"] = float(cm[0, 0])
                metrics["false_negatives"] = float(cm[1, 0])

            # Multiclass metrics
            if is_multiclass:
                metrics["accuracy_balanced"] = sk_metrics.balanced_accuracy_score(targets, predictions)
            
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
    predictions: Union[Sequence[float], List[float]],
    targets: Union[Sequence[float], List[float]],
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
    
    # Basic metrics
    errors = [p - t for p, t in zip(predictions, targets)]
    abs_errors = [abs(e) for e in errors]
    squared_errors = [e * e for e in errors]

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    mse = sum(squared_errors) / len(squared_errors) if squared_errors else 0.0
    rmse = math.sqrt(mse)
    
    # Try to use sklearn for advanced metrics if available
    if HAS_SKLEARN:
        try:
            r2 = sk_metrics.r2_score(targets, predictions)
            explained_variance = sk_metrics.explained_variance_score(targets, predictions)
            return {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "explained_variance": explained_variance,
            }
        except Exception as e:
            logger.warning(f"Error computing sklearn metrics: {e}")
    
    # Compute R2 manually
    if len(targets) > 1:
        mean_target = sum(targets) / len(targets)
        total_variance = sum((t - mean_target) ** 2 for t in targets)
        if total_variance > 0:
            r2 = 1.0 - (sum(squared_errors) / total_variance)
        else:
            r2 = 0.0
    else:
        r2 = 0.0
    
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def compute_text_generation_metrics(
    predictions: Union[Sequence[str], List[str]],
    references: Union[Sequence[List[str]], List[List[str]]],
    latencies_ms: Optional[Union[Sequence[float], List[float]]] = None,
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
        if any(pred.strip() == ref.strip() for ref in refs):
            exact_matches += 1
    
    exact_match_rate = exact_matches / len(predictions)
    
    metrics = {
        "exact_match": exact_match_rate,
    }
    
    # Try to compute BLEU score if nltk is available
    if HAS_NLTK:
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.tokenize import word_tokenize
            
            # Ensure tokenizers are available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                try:
                    nltk.download('punkt', quiet=True)
                except Exception as e:
                    logger.warning(f"Failed to download NLTK punkt tokenizer: {e}")
                    # Fall back to simple tokenization
                    word_tokenize = lambda text: text.lower().split()
            
            # Compute BLEU for each prediction
            smoothing = SmoothingFunction().method1
            bleu_scores = []
            
            for pred, refs in zip(predictions, references):
                try:
                    pred_tokens = word_tokenize(pred.lower())
                    refs_tokens = [word_tokenize(ref.lower()) for ref in refs]
                    
                    # Skip empty predictions or references
                    if not pred_tokens or not any(refs_tokens):
                        continue
                        
                    score = sentence_bleu(refs_tokens, pred_tokens, smoothing_function=smoothing)
                    bleu_scores.append(score)
                except Exception as e:
                    logger.warning(f"Error computing BLEU: {e}")
            
            if bleu_scores:
                metrics["bleu"] = sum(bleu_scores) / len(bleu_scores)
            else:
                metrics["bleu"] = 0.0
                
        except ImportError:
            logger.warning("nltk not available; BLEU score set to 0")
            metrics["bleu"] = 0.0
        except Exception as e:
            logger.warning(f"Error in BLEU calculation: {e}")
            metrics["bleu"] = 0.0
    else:
        metrics["bleu"] = 0.0
    
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
    
    # Add latency metrics if provided
    if latencies_ms:
        if len(latencies_ms) != len(predictions):
            logger.warning(f"Latency count ({len(latencies_ms)}) doesn't match prediction count ({len(predictions)})")
        
        latencies = sorted(latencies_ms)
        avg_latency = sum(latencies) / len(latencies)
        
        metrics.update({
            "latency_ms_avg": avg_latency,
            "latency_ms_p50": _percentile(latencies, 50),
            "latency_ms_p90": _percentile(latencies, 90),
            "latency_ms_p99": _percentile(latencies, 99),
        })
    
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