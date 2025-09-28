import sys
import types
from typing import Any

import pytest
from pathlib import Path

pytest.importorskip("pydantic")
pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TelemetryConfig
from src.telemetry import ExperimentLogger


class DummyConfig(dict):
    def update(self, other: Any, allow_val_change: bool = True):
        super().update(other)


class DummyRun:
    def __init__(self):
        self.logged = []
        self.artifacts = []
        self.summary = {}
        self.config = DummyConfig()
        self.name = "dummy"

    def log(self, payload):
        self.logged.append(payload)

    def log_artifact(self, artifact):
        self.artifacts.append(artifact.name)

    def finish(self):
        pass


class DummyArtifact:
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type
        self.files = []

    def add_file(self, path: str):
        self.files.append(path)


created_writers = []


class DummySummaryWriter:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.scalars = []
        self.texts = []
        created_writers.append(self)

    def add_scalar(self, key: str, value: float, step: int):
        self.scalars.append((key, value, step))

    def add_text(self, key: str, text: str):
        self.texts.append((key, text))

    def flush(self):
        pass

    def close(self):
        pass


@pytest.fixture
def fake_wandb(monkeypatch):
    run = DummyRun()
    module = types.SimpleNamespace(
        init=lambda **kwargs: run,
        Artifact=DummyArtifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", module)
    return run


@pytest.fixture
def fake_summary_writer(monkeypatch, tmp_path):
    from torch.utils import tensorboard

    monkeypatch.setattr(tensorboard, "SummaryWriter", DummySummaryWriter)
    return tmp_path


def test_experiment_logger_full_stack(fake_wandb, fake_summary_writer, tmp_path, monkeypatch):
    cfg = TelemetryConfig(
        enable_wandb=True,
        wandb_project="granite-tests",
        enable_tensorboard=True,
        tb_log_dir=str(fake_summary_writer / "logs"),
        wandb_tags=["unit"],
    )
    snapshot = {"model": {"type": "demo"}}

    artifact_path = tmp_path / "artifact.txt"
    artifact_path.write_text("demo")

    with ExperimentLogger(cfg, snapshot) as logger:
        logger.log_metrics(1, loss=0.5)
        logger.log_params(batch_size=4)
        logger.log_artifact(artifact_path, name="artifact")
        logger.set_summary(final_loss=0.5)

    assert fake_wandb.logged
    assert "artifact" in fake_wandb.artifacts
    assert created_writers
    assert any(entry[0] == "loss" for entry in created_writers[0].scalars)
