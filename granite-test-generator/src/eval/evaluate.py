"""Evaluation loop helpers integrated with the experiment telemetry stack."""

from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - optional torch dependency
    import torch
except Exception:  # pragma: no cover - fallback when torch unavailable
    torch = None  # type: ignore

from src.telemetry import ExperimentLogger

from .metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_text_generation_metrics,
)

LOGGER = logging.getLogger(__name__)


def evaluate(
    model: Any,
    dataloader: Iterable,
    task_type: str,
    device: Optional[str] = None,
    experiment_logger: Optional[ExperimentLogger] = None,
    output_dir: Path | str = Path("artifacts/eval"),
    epoch: Optional[int] = None,
) -> Dict[str, float]:
    """Run model evaluation and persist metrics and artifacts."""
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


def _flatten(value: Any) -> list[Any]:
    """Flatten nested containers and tensors into a simple list."""
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().flatten().tolist()
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
