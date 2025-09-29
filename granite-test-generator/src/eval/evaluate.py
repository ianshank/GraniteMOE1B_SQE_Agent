"""
Evaluation utilities for ML models.

This module provides functions for evaluating ML models on various tasks
and computing metrics.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np

from .metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_text_generation_metrics,
)

logger = logging.getLogger(__name__)


def evaluate(
    model: Any,
    task_type: str,
    predictions: Optional[Union[List[Any], np.ndarray]] = None,
    targets: Optional[Union[List[Any], np.ndarray]] = None,
    input_data: Optional[Union[List[Any], np.ndarray]] = None,
    batch_size: int = 16,
    telemetry: Optional[Any] = None,
    checkpoint_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    step: int = 0,
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset and compute metrics.

    Args:
        model: Model to evaluate
        task_type: Task type (classification, regression, text_generation)
        predictions: Optional pre-computed predictions
        targets: Target values
        input_data: Optional input data for computing predictions
        batch_size: Batch size for inference
        telemetry: Optional telemetry configuration
        checkpoint_path: Optional path to model checkpoint to log as artifact
        output_dir: Optional directory to save evaluation results
        step: Step or epoch number for logging

    Returns:
        Dictionary of evaluation metrics
    """
    if task_type not in ("classification", "regression", "text_generation"):
        raise ValueError(f"Unsupported task type: {task_type}")

    # Compute predictions if not provided
    if predictions is None and input_data is not None and model is not None:
        predictions = _forward(model, input_data, batch_size)

    if predictions is None or targets is None:
        raise ValueError("Either predictions or input_data and targets must be provided")

    # Flatten predictions and targets if they are tensors or arrays
    predictions = _flatten(predictions)
    targets = _flatten(targets)

    # Compute metrics based on task type
    if task_type == "classification":
        # Check if probabilities are available (for binary classification)
        probabilities = None
        if hasattr(predictions, "shape") and len(predictions.shape) > 1:
            # Multi-class probabilities, take argmax for predictions
            pred_classes = np.argmax(predictions, axis=1).tolist()
            # For binary classification, extract positive class probability
            if predictions.shape[1] == 2:
                probabilities = predictions[:, 1].tolist()
            metrics = compute_classification_metrics(pred_classes, targets, probabilities)
        else:
            # Already class indices
            metrics = compute_classification_metrics(predictions, targets)

    elif task_type == "regression":
        metrics = compute_regression_metrics(predictions, targets)

    elif task_type == "text_generation":
        # For text generation, targets should be a list of lists (multiple references)
        if not isinstance(targets[0], list):
            targets = [[t] for t in targets]
        metrics = compute_text_generation_metrics(predictions, targets)

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Add task type to metrics
    metrics["task_type"] = task_type

    # Save evaluation report if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_path = output_path / "eval_report.json"
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved evaluation report to {report_path}")

    # Log metrics to telemetry if provided
    if telemetry:
        try:
            from src.telemetry import ExperimentLogger

            with ExperimentLogger(telemetry) as logger:
                # Log metrics
                logger.log_metrics(step, **metrics)
                logger.set_summary(**{f"eval_{k}": v for k, v in metrics.items() if _is_number(v)})

                # Log checkpoint if provided
                if checkpoint_path:
                    logger.log_artifact(checkpoint_path, name="training-checkpoint")

                # Log evaluation report if saved
                if output_dir:
                    report_path = Path(output_dir) / "eval_report.json"
                    if report_path.exists():
                        logger.log_artifact(report_path, name="evaluation-report")
        except ImportError:
            logger.warning("Telemetry module not available, skipping telemetry logging")
        except Exception as e:
            logger.warning(f"Failed to log evaluation metrics to telemetry: {e}")

    return metrics


def _split_batch(data: Union[List[Any], np.ndarray], batch_size: int) -> List[Any]:
    """Split data into batches."""
    if hasattr(data, "shape"):
        # NumPy array or tensor
        num_samples = data.shape[0]
    else:
        # List or other sequence
        num_samples = len(data)

    return [
        data[i : i + batch_size] for i in range(0, num_samples, batch_size)
    ]


def _forward(model: Any, data: Union[List[Any], np.ndarray], batch_size: int) -> List[Any]:
    """Forward pass through the model in batches."""
    batches = _split_batch(data, batch_size)
    all_preds = []

    for batch in batches:
        # Try different forward methods depending on model type
        try:
            if hasattr(model, "predict"):
                preds = model.predict(batch)
            elif hasattr(model, "forward"):
                preds = model.forward(batch)
            elif callable(model):
                preds = model(batch)
            else:
                raise ValueError("Model has no predict or forward method")

            # Convert to list if tensor or array
            if hasattr(preds, "detach"):
                preds = preds.detach().cpu().numpy()
            elif hasattr(preds, "numpy"):
                preds = preds.numpy()

            all_preds.extend(preds)
        except Exception as e:
            logger.error(f"Error during model forward pass: {e}")
            raise

    return all_preds


def _flatten(data: Union[List[Any], np.ndarray]) -> List[Any]:
    """Flatten data to a Python list."""
    if data is None:
        return []

    # Convert tensor to numpy if possible
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    elif hasattr(data, "numpy"):
        data = data.numpy()

    # Convert numpy array to list
    if isinstance(data, np.ndarray):
        return data.tolist()

    return list(data)


def _flatten_probabilities(probs: Union[List[Any], np.ndarray]) -> Tuple[List[int], List[float]]:
    """Flatten probability outputs to class predictions and probabilities."""
    if probs is None:
        return [], []

    # Convert tensor to numpy if possible
    if hasattr(probs, "detach"):
        probs = probs.detach().cpu().numpy()
    elif hasattr(probs, "numpy"):
        probs = probs.numpy()

    # Multi-class case
    if isinstance(probs, np.ndarray) and len(probs.shape) > 1:
        # Get class predictions
        preds = np.argmax(probs, axis=1).tolist()
        # For binary classification, extract positive class probability
        if probs.shape[1] == 2:
            probs = probs[:, 1].tolist()
        else:
            # For multi-class, we don't have a single probability value
            probs = None
    else:
        # Binary case with single probability value
        preds = [1 if p >= 0.5 else 0 for p in probs]
        probs = list(probs)

    return preds, probs


def _is_number(value: Any) -> bool:
    """Check if a value is a number."""
    if isinstance(value, (int, float)):
        return not isinstance(value, bool) and not math.isnan(value) if hasattr(math, "isnan") else True
    return False
