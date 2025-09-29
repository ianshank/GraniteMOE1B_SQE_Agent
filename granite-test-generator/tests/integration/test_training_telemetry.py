import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("torch")
pytest.importorskip("dotenv")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from train import run_training


class _IntegrationRun:
    def __init__(self):
        self.logged = []
        self.artifacts = []
        self.summary = {}
        self.config = {}
        self.name = "integration"

    def log(self, payload, step=None):
        self.logged.append(payload)

    def log_artifact(self, artifact):
        self.artifacts.append(artifact.name)

    def finish(self):
        pass


class _IntegrationArtifact:
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type
        self._files = []

    def add_file(self, path: str):
        self._files.append(path)


created_writers = []


class _IntegrationWriter:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.scalars = []
        created_writers.append(self)

    def add_scalar(self, key: str, value: float, step: int):
        self.scalars.append((key, value, step))

    def add_text(self, key: str, value: str):
        pass

    def flush(self):
        pass

    def close(self):
        pass


@pytest.fixture
def stub_wandb(monkeypatch):
    run = _IntegrationRun()
    module = types.SimpleNamespace(init=lambda **kwargs: run, Artifact=_IntegrationArtifact)
    monkeypatch.setitem(sys.modules, "wandb", module)
    return run


@pytest.fixture
def stub_summary_writer(monkeypatch):
    from torch.utils import tensorboard

    monkeypatch.setattr(tensorboard, "SummaryWriter", _IntegrationWriter)
    return _IntegrationWriter


def test_training_generates_eval_report(tmp_path):
    out_dir = tmp_path / "baseline"
    run_training([
        "--output-dir", str(out_dir),
        "--task-type", "classification",
        "--epochs", "1",
        "--batch-size", "4",
    ])
    report_path = out_dir / "eval" / "eval_report.json"
    assert report_path.exists()
    data = report_path.read_text()
    assert "accuracy" in data


def test_training_with_telemetry(stub_wandb, stub_summary_writer, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    out_dir = tmp_path / "telemetry"
    tb_dir = tmp_path / "tb"

    run_training([
        "--output-dir", str(out_dir),
        "--task-type", "classification",
        "--epochs", "1",
        "--batch-size", "4",
        "--enable-wandb",
        "--wandb-project", "integration-tests",
        "--enable-tensorboard",
        "--tb-log-dir", str(tb_dir),
        "--log-checkpoints",
    ])

    # Just check that the files were created, since we're using a stub
    checkpoint = out_dir / "checkpoints" / "model.pt"
    assert checkpoint.exists()
    report_path = out_dir / "eval" / "eval_report.json"
    assert report_path.exists()
