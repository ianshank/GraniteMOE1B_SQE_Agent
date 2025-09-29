"""Evaluation loop helpers integrated with the experiment telemetry stack."""

from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Union, List, Tuple

try:  # pragma: no cover - optional torch dependency
    import torch
except ImportError:  # pragma: no cover - fallback when torch unavailable
    torch = None  # type: ignore

try:  # pragma: no cover - optional numpy dependency
    import numpy as np
except ImportError:  # pragma: no cover - fallback when numpy unavailable
    np = None  # type: ignore

from src.telemetry import ExperimentLogger

from .metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_text_generation_metrics,
)

LOGGER = logging.getLogger(__name__)


def evaluate(
    model: Any,
    dataloader: Optional[Iterable] = None,
    task_type: str = "classification",
    device: Optional[str] = None,
    experiment_logger: Optional[ExperimentLogger] = None,
    output_dir: Union[Path, str] = Path("artifacts/eval"),
    epoch: Optional[int] = None,
    predictions: Optional[Union[List[Any], np.ndarray]] = None,
    targets: Optional[Union[List[Any], np.ndarray]] = None,
    input_data: Optional[Union[List[Any], np.ndarray]] = None,
    batch_size: int = 16,
    telemetry: Optional[Any] = None,
    checkpoint_path: Optional[str] = None,
    step: int = 0,
) -> Dict[str, float]:
    """
    Run model evaluation and persist metrics and artifacts.
    
    Args:
        model: Model to evaluate
        dataloader: Dataloader with evaluation data (for dataloader-based evaluation)
        task_type: Type of task ('classification', 'regression', or 'text')
        device: Device to run evaluation on ('cpu' or 'cuda')
        experiment_logger: Optional experiment logger for telemetry
        output_dir: Directory to save evaluation artifacts
        epoch: Optional epoch number for logging
        predictions: Optional pre-computed predictions (for direct evaluation)
        targets: Target values (for direct evaluation)
        input_data: Optional input data for computing predictions
        batch_size: Batch size for inference
        telemetry: Optional telemetry configuration (legacy parameter)
        checkpoint_path: Optional path to model checkpoint to log as artifact
        step: Step or epoch number for logging
        
    Returns:
        Dictionary of evaluation metrics
    """
    if task_type not in ("classification", "regression", "text", "text_generation"):
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Handle legacy text_generation -> text mapping
    if task_type == "text_generation":
        task_type = "text"
    
    # Determine evaluation mode
    if dataloader is not None:
        # Dataloader-based evaluation (newer approach)
        return _evaluate_with_dataloader(
            model, dataloader, task_type, device, experiment_logger, output_dir, epoch
        )
    elif predictions is not None and targets is not None:
        # Direct evaluation with pre-computed predictions
        return _evaluate_direct(
            model, task_type, predictions, targets, input_data, batch_size, 
            telemetry, checkpoint_path, output_dir, step
        )
    else:
        raise ValueError("Either dataloader or predictions+targets must be provided")


def _evaluate_with_dataloader(
    model: Any,
    dataloader: Iterable,
    task_type: str,
    device: Optional[str] = None,
    experiment_logger: Optional[ExperimentLogger] = None,
    output_dir: Union[Path, str] = Path("artifacts/eval"),
    epoch: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate using a dataloader (newer approach)."""
    device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    if torch is not None and hasattr(model, "to"):
        model = model.to(device)  # type: ignore[assignment]
    if hasattr(model, "eval"):
        model.eval()  # type: ignore[attr-defined]
    results: MutableMapping[str, float] = {}

    predictions: list[Any] = []
    targets: list[Any] = []
    probabilities: list[Any] = []
    latencies_ms: list[float] = []

    forward_ctx = torch.no_grad if torch is not None else nullcontext  # type: ignore[assignment]

    for step, batch in enumerate(dataloader, start=1):
        try:
            inputs, labels = _split_batch(batch)
            start = time.perf_counter()
            with forward_ctx():  # type: ignore[operator]
                preds = _forward(model, inputs, device)
            latency_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(latency_ms)
            LOGGER.debug("Eval step %s latency %.3fms", step, latency_ms)

            if isinstance(preds, tuple):
                primary, secondary = preds
            else:
                primary, secondary = preds, None

            primary_items = _flatten(primary)
            predictions.extend(primary_items)

            if secondary is not None:
                secondary_items = _flatten_probabilities(secondary, len(primary_items))
                probabilities.extend(secondary_items)

            targets.extend(_flatten(labels))
        except Exception as e:
            LOGGER.error(f"Error in evaluation step {step}: {e}")
            continue

    try:
        if task_type == "classification":
            probs_iter = probabilities if probabilities else None
            metrics = compute_classification_metrics(predictions, targets, probs_iter)
        elif task_type == "regression":
            metrics = compute_regression_metrics(predictions, targets)
        elif task_type == "text":
            metrics = compute_text_generation_metrics(
                [str(pred) for pred in predictions],
                [[str(label)] if not isinstance(label, (list, tuple)) else label for label in targets],
                latencies_ms=latencies_ms,
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        if task_type != "text":
            metrics["latency_ms_avg"] = float(sum(latencies_ms) / max(len(latencies_ms), 1))
            metrics["latency_ms_p95"] = float(_percentile(latencies_ms, 95))

        results.update({key: float(value) for key, value in metrics.items() if _is_number(value)})
    except ValueError as e:
        LOGGER.warning(f"Error computing metrics: {e}")
        results = {"error": 1.0}

    step_index = epoch if epoch is not None else len(predictions)
    if experiment_logger is not None:
        experiment_logger.log_metrics(step_index, **results)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    LOGGER.info("Saved evaluation report to %s", report_path)

    if experiment_logger is not None:
        experiment_logger.log_artifact(report_path, name="eval_report")
        experiment_logger.set_summary(**results)

    return dict(results)


def _evaluate_direct(
    model: Any,
    task_type: str,
    predictions: Union[List[Any], np.ndarray],
    targets: Union[List[Any], np.ndarray],
    input_data: Optional[Union[List[Any], np.ndarray]] = None,
    batch_size: int = 16,
    telemetry: Optional[Any] = None,
    checkpoint_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    step: int = 0,
) -> Dict[str, float]:
    """Evaluate with direct predictions and targets (legacy approach)."""
    # Compute predictions if not provided
    if predictions is None and input_data is not None and model is not None:
        predictions = _forward_batch(model, input_data, batch_size)

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

    elif task_type == "text":
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
        LOGGER.info(f"Saved evaluation report to {report_path}")

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
            LOGGER.warning("Telemetry module not available, skipping telemetry logging")
        except Exception as e:
            LOGGER.warning(f"Failed to log evaluation metrics to telemetry: {e}")

    return metrics


def _split_batch(batch: Any) -> tuple[Any, Any]:
    """Split supported batch structures into (inputs, labels)."""
    if isinstance(batch, Mapping):
        labels = batch.get("labels") or batch.get("targets")
        inputs = batch.get("inputs") or batch.get("features")
        if inputs is None or labels is None:
            raise ValueError("Batch mapping must contain 'inputs'/'labels' keys")
        return inputs, labels
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1]
    raise TypeError("Unsupported batch structure for evaluation")


def _forward(model: Any, inputs: Any, device: str) -> Any:
    """Execute a forward pass for arbitrary model call styles."""
    if hasattr(model, "eval_step"):
        return model.eval_step(inputs, device=device)
    if callable(model):
        if torch is not None and isinstance(inputs, torch.Tensor):
            return model(inputs.to(device))
        return model(inputs)
    raise TypeError("Model must implement 'eval_step' or be callable")


def _forward_batch(model: Any, data: Union[List[Any], np.ndarray], batch_size: int) -> List[Any]:
    """Forward pass through the model in batches."""
    batches = _split_batch_data(data, batch_size)
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
            LOGGER.error(f"Error during model forward pass: {e}")
            raise

    return all_preds


def _split_batch_data(data: Union[List[Any], np.ndarray], batch_size: int) -> List[Any]:
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


def _flatten(value: Any) -> list[Any]:
    """Flatten nested containers and tensors into a simple list."""
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().flatten().tolist()
    if np is not None and isinstance(value, np.ndarray):
        return value.flatten().tolist()
    if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [value]


def _flatten_probabilities(value: Any, expected_length: int) -> list[Any]:
    """Flatten probability-like outputs while respecting sample counts."""
    if torch is not None and isinstance(value, torch.Tensor):
        array = value.detach().cpu().flatten().tolist()
        if len(array) == expected_length:
            return array
        if len(array) == 2:
            return [array[1]]
        return array
    if isinstance(value, (list, tuple)):
        if len(value) == expected_length and not any(isinstance(item, (list, tuple)) for item in value):
            return list(value)
        if len(value) == 2 and not isinstance(value[1], (list, tuple)):
            return [float(value[1])]  # type: ignore[arg-type]
        return [list(item) if isinstance(item, (list, tuple)) else item for item in value]
    return [value]


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return the percentile using linear interpolation for small arrays."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (percentile / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return float(ordered[int(k)])
    d0 = ordered[f] * (c - k)
    d1 = ordered[c] * (k - f)
    return float(d0 + d1)


def _is_number(value: Any) -> bool:
    """Safely determine whether a value is numeric."""
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False