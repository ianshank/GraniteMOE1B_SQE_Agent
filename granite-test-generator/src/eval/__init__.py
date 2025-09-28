"""Evaluation utilities package with lazy imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_text_generation_metrics,
)

if TYPE_CHECKING:  # pragma: no cover
    from .evaluate import evaluate as evaluate_fn


def evaluate(*args: Any, **kwargs: Any):
    from .evaluate import evaluate as _evaluate

    return _evaluate(*args, **kwargs)


__all__ = [
    "compute_classification_metrics",
    "compute_regression_metrics",
    "compute_text_generation_metrics",
    "evaluate",
]
