"""Unit tests for sanitized logging in GraniteMoETrainer."""

import logging
from typing import Dict, Any

import pytest

from src.models.granite_moe import GraniteMoETrainer


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records = []

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        self.records.append(record)


def test_granite_moe_trainer_sanitized_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Ensure only safe keys are logged and secrets are not leaked."""
    logger = logging.getLogger("src.models.granite_moe")
    handler = _CaptureHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    config: Dict[str, Any] = {
        "model_name": "foo/bar",
        "epochs": 1,
        "batch_size": 2,
        "output_dir": "models",
        "WANDB_API_KEY": "should-not-appear",
    }

    GraniteMoETrainer(config)

    # Verify a record was logged with sanitized content
    assert any(
        "sanitized" in r.getMessage() and "should-not-appear" not in r.getMessage()
        for r in handler.records
    )


