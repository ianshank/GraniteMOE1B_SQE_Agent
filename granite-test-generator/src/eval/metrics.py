"""Reusable evaluation metrics for Granite SQE experiments."""

from __future__ import annotations

import logging
import math
from statistics import mean
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - torch optional in CI
    import torch
except Exception:  # pragma: no cover - fallback when torch unavailable
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import evaluate as hf_evaluate
except Exception:  # pragma: no cover - evaluation optional
    hf_evaluate = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def compute_classification_metrics(
    predictions: Iterable,
    targets: Iterable,
    probabilities: Optional[Iterable] = None,
) -> Mapping[str, float]:
    """Compute standard classification metrics with macro/micro variants."""
    preds = _flatten_scalar(predictions)
    gold = _flatten_scalar(targets)
    if len(preds) != len(gold):
        raise ValueError("Predictions and targets must share the same length")

    labels = sorted(set(preds) | set(gold))
    tp = {label: 0 for label in labels}
    fp = {label: 0 for label in labels}
    fn = {label: 0 for label in labels}

    for pred, true in zip(preds, gold):
        if pred == true:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    precision_scores: List[float] = []
    recall_scores: List[float] = []
    f1_scores: List[float] = []
    for label in labels:
        tp_val = tp[label]
        fp_val = fp[label]
        fn_val = fn[label]
        precision = tp_val / (tp_val + fp_val) if (tp_val + fp_val) else 0.0
        recall = tp_val / (tp_val + fn_val) if (tp_val + fn_val) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    accuracy = sum(1 for pred, true in zip(preds, gold) if pred == true) / max(len(gold), 1)
    metrics: MutableMapping[str, float] = {
        "accuracy": accuracy,
        "precision_macro": mean(precision_scores) if precision_scores else 0.0,
        "recall_macro": mean(recall_scores) if recall_scores else 0.0,
        "f1_macro": mean(f1_scores) if f1_scores else 0.0,
    }

    tp_total = sum(tp.values())
    fp_total = sum(fp.values())
    fn_total = sum(fn.values())
    precision_micro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    recall_micro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f1_micro = (2 * precision_micro * recall_micro / (precision_micro + recall_micro)) if (precision_micro + recall_micro) else 0.0
    metrics.update(
        {
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
        }
    )

    if probabilities is not None:
        scores = _normalise_probabilities(probabilities, len(preds))
        if scores:
            metrics["auroc"] = _binary_auroc(gold, scores)
    return metrics


def compute_regression_metrics(predictions: Iterable, targets: Iterable) -> Mapping[str, float]:
    """Compute MAE, RMSE, and R^2 regression metrics."""
    preds = _flatten_float(predictions)
    gold = _flatten_float(targets)
    if len(preds) != len(gold):
        raise ValueError("Predictions and targets must share the same length")

    n = len(preds)
    if n == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}

    diffs = [p - g for p, g in zip(preds, gold)]
    mae = sum(abs(d) for d in diffs) / n
    rmse = math.sqrt(sum(d * d for d in diffs) / n)
    mean_gold = sum(gold) / n
    ss_res = sum((p - g) ** 2 for p, g in zip(preds, gold))
    ss_tot = sum((g - mean_gold) ** 2 for g in gold)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def compute_text_generation_metrics(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]] | Sequence[str],
    latencies_ms: Optional[Sequence[float]] = None,
) -> Mapping[str, float]:
    """Compute text generation metrics with optional HF evaluate integration."""
    refs = [_normalise_references(ref) for ref in references]
    metrics: MutableMapping[str, float] = {}

    if hf_evaluate is not None:
        try:
            bleu = hf_evaluate.load("bleu")
            rouge = hf_evaluate.load("rouge")
            chrf = hf_evaluate.load("chrf")
            metrics["bleu"] = float(bleu.compute(predictions=predictions, references=refs)["bleu"])
            metrics["rouge_l"] = float(rouge.compute(predictions=predictions, references=refs)["rougeL"])
            metrics["chrf"] = float(chrf.compute(predictions=predictions, references=refs)["score"])
        except Exception as exc:  # pragma: no cover - network/cache issues
            LOGGER.warning("Failed to compute advanced text metrics: %s", exc)

    exact = []
    for pred, ref_list in zip(predictions, refs):
        exact.append(1.0 if pred.strip() in {ref.strip() for ref in ref_list} else 0.0)
    metrics["exact_match"] = mean(exact) if exact else 0.0

    if latencies_ms:
        metrics["latency_ms_avg"] = mean(latencies_ms)
        metrics["latency_ms_p95"] = _percentile(latencies_ms, 95)

    return metrics


def _flatten_scalar(values: Iterable) -> List[int]:
    """Normalise arbitrary iterables of scalar-like values into integers."""
    flattened: List[int] = []
    for value in _iterate(values):
        flattened.append(int(round(float(value))))
    return flattened


def _flatten_float(values: Iterable) -> List[float]:
    """Normalise arbitrary iterables of scalar-like values into floats."""
    return [float(item) for item in _iterate(values)]


def _iterate(values: Iterable) -> List[float]:
    """Return a flattened list representation for nested tensor-friendly inputs."""
    if torch is not None and isinstance(values, torch.Tensor):
        return values.detach().cpu().flatten().tolist()
    if isinstance(values, (list, tuple)):
        output: List[float] = []
        for item in values:
            output.extend(_iterate(item))
        return output
    return [float(values)]


def _normalise_probabilities(probabilities: Iterable, expected_length: int) -> List[float]:
    """Extract probability scores suitable for AUROC computation."""
    scores: List[float] = []
    for item in probabilities:
        if torch is not None and isinstance(item, torch.Tensor):
            normalised = item.detach().cpu().flatten().tolist()
        elif isinstance(item, (list, tuple)):
            normalised = list(item)
        else:
            normalised = [item]
        if len(normalised) == 1:
            scores.append(float(normalised[0]))
        elif len(normalised) == 2:
            scores.append(float(normalised[1]))
        else:
            LOGGER.debug("Skipping AUROC probability vector of length %s", len(normalised))
    if len(scores) != expected_length:
        LOGGER.debug("Probability count %s does not match expected %s", len(scores), expected_length)
        return []
    return scores


def _binary_auroc(targets: List[int], scores: List[float]) -> float:
    """Compute AUROC for binary classification without external dependencies."""
    if len(set(targets)) < 2:
        LOGGER.debug("AUROC undefined for single-class targets")
        return float("nan")
    positive_label = max(targets)
    pairs = sorted(zip(scores, targets), key=lambda x: x[0])
    pos_total = sum(1 for target in targets if target == positive_label)
    neg_total = len(targets) - pos_total
    if pos_total == 0 or neg_total == 0:
        LOGGER.debug("AUROC undefined due to missing positive or negative samples")
        return float("nan")

    tpr = [0.0]
    fpr = [0.0]
    tp = fp = 0
    for score, label in reversed(pairs):
        if label == positive_label:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / pos_total)
        fpr.append(fp / neg_total)

    auc = 0.0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return auc


def _normalise_references(ref: Sequence[str] | str) -> List[str]:
    """Ensure text-generation references are represented as simple string lists."""
    if isinstance(ref, str):
        return [ref]
    return [str(item) for item in ref]


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Compute the percentile using linear interpolation for small samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return float(ordered[int(k)])
    d0 = ordered[f] * (c - k)
    d1 = ordered[c] * (k - f)
    return float(d0 + d1)
