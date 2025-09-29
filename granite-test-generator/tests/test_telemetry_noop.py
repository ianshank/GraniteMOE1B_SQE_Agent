import logging
import sys
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TelemetryConfig
from src.telemetry import ExperimentLogger


def test_experiment_logger_noop(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    cfg = TelemetryConfig()
    snapshot = {"model": {"type": "noop"}}

    with ExperimentLogger(cfg, snapshot) as logger:
        logger.log_metrics(1, loss=0.01)
        logger.log_params(learning_rate=1e-3)
        logger.set_summary(final_loss=0.01)
        logger.log_artifact(tmp_path / "missing.txt")

    runs_dir = Path("runs")
    assert not runs_dir.exists() or not any(runs_dir.iterdir())
