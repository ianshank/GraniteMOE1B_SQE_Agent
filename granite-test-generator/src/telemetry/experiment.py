"""Unified experiment logging for W&B and TensorBoard."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import AbstractContextManager
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from src.config.telemetry import TelemetryConfig

LOGGER = logging.getLogger(__name__)


class ExperimentLogger(AbstractContextManager["ExperimentLogger"]):
    """Coordinate Weights & Biases and TensorBoard logging with graceful degradation."""

    def __init__(
        self,
        telemetry_cfg: TelemetryConfig,
        config_snapshot: Optional[Mapping[str, Any]] = None,
        run_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the experiment logger.
        
        Args:
            telemetry_cfg: Configuration for telemetry
            config_snapshot: Configuration snapshot to log
            run_name: Optional explicit run name
        """
        self._cfg = telemetry_cfg
        self._config_snapshot = dict(config_snapshot) if config_snapshot else {}
        self._run_name = run_name
        self._wandb_run: Optional[Any] = None
        self._tb_writer: Optional[Any] = None
        self._start_time: Optional[float] = None
        self._git_sha: Optional[str] = None
        self._closed = False

    def __enter__(self) -> "ExperimentLogger":
        self.start_run()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()

    def start_run(self) -> None:
        """Start the experiment run, initializing loggers."""
        if self._start_time is not None:
            return
        self._start_time = time.time()
        self._run_name = self._run_name or self._derive_run_name()
        self._git_sha = self._detect_git_sha()
        LOGGER.debug("Starting experiment run: %s", self._run_name)

        if self._cfg.enable_wandb:
            self._start_wandb_run()
        else:
            LOGGER.debug("W&B telemetry disabled by configuration")

        if self._cfg.enable_tensorboard:
            self._start_tensorboard_writer()
        else:
            LOGGER.debug("TensorBoard telemetry disabled by configuration")

        if self._config_snapshot:
            self.log_params(**self._config_snapshot)

    def log_metrics(self, step: int, **metrics: float) -> None:
        """
        Log metrics at a specific step.
        
        Args:
            step: Training step or epoch
            **metrics: Metrics to log as key-value pairs
        """
        if not metrics:
            return
        filtered = {k: float(v) for k, v in metrics.items() if self._is_number(v)}
        if not filtered:
            return
        LOGGER.debug("Logging metrics at step %s: %s", step, filtered)

        if self._wandb_run is not None:
            payload = dict(filtered)
            payload["step"] = step
            try:
                self._wandb_run.log(payload)
            except Exception as exc:  # pragma: no cover - defensive safety
                LOGGER.warning("Failed to log metrics to W&B: %s", exc)

        if self._tb_writer is not None:
            for key, value in filtered.items():
                try:
                    self._tb_writer.add_scalar(key, value, step)
                except Exception as exc:  # pragma: no cover - defensive safety
                    LOGGER.warning("Failed to write TensorBoard metric %s: %s", key, exc)
            self._tb_writer.flush()

    def log_params(self, **params: Any) -> None:
        """
        Log parameters for the experiment.
        
        Args:
            **params: Parameters to log as key-value pairs
        """
        if not params:
            return
        LOGGER.debug("Logging parameters: %s", params)
        if self._wandb_run is not None:
            try:
                config = getattr(self._wandb_run, "config", None)
                if config is not None:
                    config.update(params, allow_val_change=True)
            except Exception as exc:  # pragma: no cover - defensive safety
                LOGGER.warning("Failed to update W&B config: %s", exc)

        if self._tb_writer is not None:
            try:
                self._tb_writer.add_text("hyperparameters", json.dumps(params, indent=2))
            except Exception as exc:  # pragma: no cover - defensive safety
                LOGGER.warning("Failed to write TensorBoard params: %s", exc)

    def log_artifact(self, path: Any, name: Optional[str] = None, type: str = "artifact") -> None:
        """
        Log an artifact file.
        
        Args:
            path: Path to the artifact file
            name: Optional name for the artifact
            type: Type of artifact (e.g., "model", "dataset")
        """
        if not self._cfg.log_artifacts:
            LOGGER.info(f"Artifact logging disabled, skipping: {path}")
            return

        file_path = Path(path)
        if not file_path.exists():
            LOGGER.warning("Artifact path %s does not exist; skipping upload", file_path)
            return

        artifact_name = name or file_path.name

        if self._wandb_run is not None:
            try:
                import wandb  # type: ignore

                artifact = wandb.Artifact(name=artifact_name, type=type)
                artifact.add_file(str(file_path))
                self._wandb_run.log_artifact(artifact)
                LOGGER.info("Logged artifact %s to W&B", file_path)
            except Exception as exc:  # pragma: no cover - defensive safety
                LOGGER.warning("Failed to log artifact to W&B: %s", exc)

    def set_summary(self, **values: Any) -> None:
        """
        Update summary metrics for the experiment.
        
        Args:
            **values: Summary metrics to set as key-value pairs
        """
        if not values:
            return
        LOGGER.debug("Updating summary metrics: %s", values)
        if self._wandb_run is not None:
            summary = getattr(self._wandb_run, "summary", None)
            if summary is not None:
                try:
                    for key, value in values.items():
                        summary[key] = value
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("Failed to update W&B summary: %s", exc)

        if self._tb_writer is not None:
            try:
                self._tb_writer.add_text("summary", json.dumps(values, indent=2))
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to write TensorBoard summary: %s", exc)

    def finish(self) -> None:
        """Finish the experiment run, cleaning up resources."""
        if self._closed:
            return
        self._closed = True
        if self._tb_writer is not None:
            try:
                self._tb_writer.flush()
                self._tb_writer.close()
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to close TensorBoard writer: %s", exc)
            self._tb_writer = None
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to close W&B run: %s", exc)
            self._wandb_run = None
        LOGGER.debug("Experiment logger finished")

    def _start_wandb_run(self) -> None:
        """Initialize the W&B run."""
        project = self._cfg.wandb_project or os.getenv("WANDB_PROJECT")
        if not project:
            LOGGER.warning("W&B enabled but no project provided; disabling W&B logging")
            return

        try:
            import wandb  # type: ignore
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("W&B library unavailable: %s", exc)
            return

        entity = self._cfg.wandb_entity or os.getenv("WANDB_ENTITY")
        run_name = self._cfg.wandb_run_name or self._cfg.wandb_name or self._run_name
        tags = list(dict.fromkeys(self._cfg.wandb_tags))
        if self._git_sha:
            tags.append(self._git_sha)
        tags = [tag for tag in tags if tag]

        os.environ.setdefault("WANDB_PROJECT", project)
        if entity:
            os.environ.setdefault("WANDB_ENTITY", entity)
        if run_name:
            os.environ.setdefault("WANDB_RUN_NAME", run_name)
        if tags:
            os.environ["WANDB_TAGS"] = ",".join(tags)

        settings = {"start_method": "thread"}
        wandb_mode = os.getenv("WANDB_MODE")
        if wandb_mode:
            LOGGER.info("W&B mode=%s", wandb_mode)

        try:
            self._wandb_run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                tags=tags or None,
                group=self._cfg.wandb_group,
                job_type=self._cfg.wandb_job_type,
                notes=self._cfg.wandb_notes,
                config=dict(self._config_snapshot),
                reinit=True,
                settings=settings,
            )
            LOGGER.info("Initialized W&B run %s", self._wandb_run.name if self._wandb_run else run_name)
            if self._git_sha and self._wandb_run is not None:
                self._wandb_run.config.update({"git_sha": self._git_sha}, allow_val_change=True)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to initialise W&B run: %s", exc)
            self._wandb_run = None

    def _start_tensorboard_writer(self) -> None:
        """Initialize the TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("TensorBoard SummaryWriter unavailable: %s", exc)
            return

        log_dir = Path(self._cfg.tb_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._tb_writer = SummaryWriter(log_dir=str(log_dir))
            LOGGER.info("TensorBoard writer initialised at %s", log_dir)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to create TensorBoard writer: %s", exc)
            self._tb_writer = None

    def _derive_run_name(self) -> str:
        """Derive a run name from the configuration snapshot."""
        # Handle model name which could be a string or a dict
        if isinstance(self._config_snapshot.get("model"), dict):
            model_name = self._config_snapshot["model"].get("type", "model")
        elif isinstance(self._config_snapshot.get("model"), str):
            model_name = self._config_snapshot["model"]
        else:
            model_name = self._config_snapshot.get("model_name", "model")
        
        # Handle dataset name which could be in different formats
        if isinstance(self._config_snapshot.get("data"), dict):
            dataset_name = self._config_snapshot["data"].get("train_file", "dataset")
        else:
            dataset_name = self._config_snapshot.get("dataset", "dataset")
        
        # Generate timestamp and create safe names
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        safe_model = str(model_name).split("/")[-1]
        safe_dataset = Path(str(dataset_name)).stem or "dataset"
        
        return f"{safe_model}-{safe_dataset}-{timestamp}"

    @staticmethod
    def _detect_git_sha() -> Optional[str]:
        """Detect the git SHA of the current repository."""
        try:
            import subprocess

            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            return sha.decode().strip()
        except Exception:  # pragma: no cover
            return None

    @staticmethod
    def _is_number(value: Any) -> bool:
        """Check if a value is a number that can be converted to float."""
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False