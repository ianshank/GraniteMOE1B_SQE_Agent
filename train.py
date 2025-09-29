"""Lightweight training harness with telemetry instrumentation."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, List, Union

from dotenv import load_dotenv

try:  # pragma: no cover - torch optional in some environments
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - graceful fallback
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore

try:
    import yaml
except Exception:  # pragma: no cover - yaml should be available
    yaml = None  # type: ignore

# Add the src directory to Python path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "granite-test-generator" / "src"))

from config import load_telemetry_from_sources
from telemetry import ExperimentLogger
from eval.evaluate import evaluate

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path("config/training_config.yaml")


@dataclass
class TrainingArtifacts:
    """Bundle of resources produced by the training loop."""

    model: Any
    metrics: Dict[str, float]
    checkpoint_path: Optional[Path]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Granite SQE training harness")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to training configuration YAML")
    parser.add_argument("--task-type", choices=["classification", "regression", "text"], default="classification")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="artifacts/training")
    parser.add_argument("--device", type=str, default=None, help="Target device override (cpu/cuda)")
    parser.add_argument("--log-checkpoints", action="store_true", help="Upload model checkpoints via telemetry backends")
    parser.add_argument("--log-interval-steps", type=int, default=None, help="Override telemetry logging interval")

    # Telemetry toggles
    parser.add_argument("--enable-wandb", dest="enable_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--disable-wandb", dest="enable_wandb", action="store_false", help="Disable W&B logging")
    parser.set_defaults(enable_wandb=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma separated list of W&B tags")

    parser.add_argument("--enable-tensorboard", dest="enable_tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--disable-tensorboard", dest="enable_tensorboard", action="store_false", help="Disable TensorBoard logging")
    parser.set_defaults(enable_tensorboard=None)
    parser.add_argument("--tb-log-dir", type=str, default=None, help="TensorBoard log directory")

    return parser.parse_args(list(argv) if argv is not None else None)


def _load_training_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        LOGGER.warning("Training config %s not found; proceeding with defaults", path)
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to load the training configuration")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_config_snapshot(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = {
        "model": config.get("model", {"type": "toy-net"}),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "task_type": args.task_type,
        },
        "data": config.get("data", {"source": "synthetic"}),
    }
    return snapshot


def _resolve_device(preferred: Optional[str]) -> str:
    if torch is None:
        return "cpu"
    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def _prepare_classification_data(batch_size: int) -> Tuple[Iterable, Iterable, int]:
    if torch is None or TensorDataset is None:
        raise RuntimeError("PyTorch is required for classification training")
    rng = torch.Generator().manual_seed(42)
    features = torch.randn(128, 4, generator=rng)
    labels = (features.sum(dim=1) > 0).long()
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=rng)
    eval_loader = DataLoader(dataset, batch_size=batch_size)
    return loader, eval_loader, features.shape[1]


def _prepare_regression_data(batch_size: int) -> Tuple[Iterable, Iterable, int]:
    if torch is None or TensorDataset is None:
        raise RuntimeError("PyTorch is required for regression training")
    rng = torch.Generator().manual_seed(24)
    features = torch.randn(128, 3, generator=rng)
    weights = torch.tensor([0.6, -0.3, 0.9])
    labels = features @ weights + 0.1
    dataset = TensorDataset(features, labels.unsqueeze(-1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=rng)
    eval_loader = DataLoader(dataset, batch_size=batch_size)
    return loader, eval_loader, features.shape[1]


def _prepare_text_data() -> Tuple[Iterable, Iterable, int]:
    samples = [
        {"inputs": "Generate summary for login flow", "labels": "Login flow summary"},
        {"inputs": "Document checkout process", "labels": "Checkout process document"},
    ]
    return samples, samples, 0


class _ClassificationNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class _RegressionNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class _TextEchoModel:
    def __call__(self, batch: Any) -> Any:
        if isinstance(batch, str):
            return batch
        if isinstance(batch, Iterable):
            return [str(item) for item in batch]
        return str(batch)


def _train_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: Any,
    device: str,
    task_type: str,
) -> Dict[str, float]:
    assert torch is not None
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0
    criterion = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()

    for batch_inputs, batch_labels in loader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        if task_type == "classification":
            loss = criterion(outputs, batch_labels)
            preds = outputs.argmax(dim=1)
            total_correct += float((preds == batch_labels).float().sum().item())
            total_examples += batch_labels.shape[0]
        else:
            loss = criterion(outputs, batch_labels)
            total_examples += batch_labels.shape[0]
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch_labels.shape[0]

    metrics = {
        "loss": total_loss / max(total_examples, 1),
    }
    if task_type == "classification":
        metrics["accuracy"] = total_correct / max(total_examples, 1)
    return metrics


def _save_checkpoint(model: nn.Module, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def run_training(argv: Optional[Iterable[str]] = None) -> TrainingArtifacts:
    try:
        load_dotenv()
    except ImportError:
        LOGGER.warning("python-dotenv is not installed; skipping loading .env file.")
    except Exception as e:
        LOGGER.warning(f"Could not load .env file: {e}")
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    config_path = Path(args.config)
    config = _load_training_config(config_path)

    telemetry_cfg = load_telemetry_from_sources(args)
    if args.log_interval_steps is not None:
        telemetry_cfg = telemetry_cfg.merged_with(log_interval_steps=args.log_interval_steps)

    device = _resolve_device(args.device)
    snapshot = _build_config_snapshot(args, config)
    snapshot["runtime"] = {"device": device}

    model: Any
    train_loader: Iterable
    eval_loader: Iterable
    input_dim = 0

    if args.task_type == "classification":
        train_loader, eval_loader, input_dim = _prepare_classification_data(args.batch_size)
        model = _ClassificationNet(input_dim)
    elif args.task_type == "regression":
        train_loader, eval_loader, input_dim = _prepare_regression_data(args.batch_size)
        model = _RegressionNet(input_dim)
    else:
        train_loader, eval_loader, _ = _prepare_text_data()
        model = _TextEchoModel()

    if torch is not None and isinstance(model, nn.Module):
        model.to(device)

    optimizer = None
    if torch is not None and isinstance(model, nn.Module):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    total_steps = 0
    best_metric = None
    best_metric_name = "accuracy" if args.task_type == "classification" else ("r2" if args.task_type == "regression" else "exact_match")

    with ExperimentLogger(telemetry_cfg, snapshot) as experiment:
        for epoch in range(1, args.epochs + 1):
            if torch is not None and isinstance(model, nn.Module):
                epoch_metrics = _train_epoch(model, train_loader, optimizer, device, args.task_type)
                total_steps += len(train_loader)
                experiment.log_metrics(total_steps, epoch=epoch, **epoch_metrics)
            else:
                LOGGER.info("Skipping parameterized training for task %s", args.task_type)

            eval_dir = Path(args.output_dir) / "eval"
            try:
                eval_metrics = evaluate(
                    model,
                    eval_loader,
                    task_type=args.task_type,
                    device=device,
                    experiment_logger=experiment,
                    output_dir=eval_dir,
                    epoch=epoch,
                )
            except ValueError as e:
                LOGGER.warning(f"Evaluation error: {e}")
                eval_metrics = {"status": "error", "message": str(e)}
                experiment.log_metrics(epoch, eval_error=1.0)
            metric_value = eval_metrics.get(best_metric_name)
            if metric_value is not None and (best_metric is None or metric_value > best_metric):
                best_metric = metric_value

        summary = {
            "best_metric": best_metric if best_metric is not None else 0.0,
            "best_metric_name": best_metric_name,
            "epochs": args.epochs,
            "total_steps": total_steps,
        }
        experiment.set_summary(**summary)

        checkpoint_path = None
        if args.log_checkpoints and torch is not None and isinstance(model, nn.Module):
            checkpoint_dir = Path(args.output_dir) / "checkpoints"
            checkpoint_path = _save_checkpoint(model, checkpoint_dir)
            experiment.log_artifact(checkpoint_path, name="training-checkpoint", type="model")

    metrics = summary if "summary" in locals() else {}
    return TrainingArtifacts(model=model, metrics=metrics, checkpoint_path=checkpoint_path)


def main(argv: Optional[Iterable[str]] = None) -> None:
    run_training(argv)


if __name__ == "__main__":
    main()