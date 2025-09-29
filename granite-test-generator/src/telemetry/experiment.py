"""
Experiment logging for ML training and evaluation.

This module provides a unified interface for logging metrics, parameters, and
artifacts to multiple backends (W&B, TensorBoard) with graceful degradation.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """
    Unified experiment logger with support for W&B and TensorBoard.

    This class provides a facade for logging metrics, parameters, and artifacts
    to multiple backends (W&B, TensorBoard) with graceful degradation when
    dependencies are not available.

    Args:
        config: TelemetryConfig instance
        config_snapshot: Optional snapshot of the full config for logging
    """

    def __init__(self, config, config_snapshot: Optional[Dict[str, Any]] = None):
        """Initialize the experiment logger."""
        self._config = config
        self._config_snapshot = config_snapshot or {}
        self._wandb_run = None
        self._tb_writer = None
        self._run_name = self._derive_run_name()

        # Initialize backends
        self._init_wandb()
        self._init_tensorboard()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
        return False  # Don't suppress exceptions

    def _derive_run_name(self) -> str:
        """Derive a run name from the config snapshot."""
        if self._config.wandb_name:
            return self._config.wandb_name

        # Try to derive from config snapshot
        model_name = "unknown"
        data_name = "unknown"

        if self._config_snapshot:
            model = self._config_snapshot.get("model")
            if model:
                if isinstance(model, dict):
                    model_name = model.get("name", model.get("type", "unknown"))
                else:
                    model_name = str(model)

            data = self._config_snapshot.get("data")
            if data:
                if isinstance(data, dict):
                    data_name = data.get("name", data.get("type", "unknown"))
                else:
                    data_name = str(data)

        return f"{model_name}-{data_name}"

    def _init_wandb(self):
        """Initialize W&B if enabled and available."""
        if not self._config.enable_wandb:
            return

        try:
            import wandb

            self._wandb_run = wandb.init(
                project=self._config.wandb_project,
                entity=self._config.wandb_entity,
                tags=self._config.wandb_tags,
                group=self._config.wandb_group,
                job_type=self._config.wandb_job_type,
                name=self._run_name,
                notes=self._config.wandb_notes,
                config=self._config_snapshot,
            )
            logger.info(
                f"W&B initialized: {self._wandb_run.name} "
                f"(project={self._config.wandb_project}, "
                f"entity={self._config.wandb_entity})"
            )
        except ImportError:
            logger.warning("W&B not installed, skipping W&B initialization")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")

    def _init_tensorboard(self):
        """Initialize TensorBoard if enabled and available."""
        if not self._config.enable_tensorboard:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(self._config.tb_log_dir, self._run_name)
            self._tb_writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard initialized: {log_dir}")
        except ImportError:
            logger.warning("TensorBoard not installed, skipping TensorBoard initialization")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")

    def log_metrics(self, step: int, **metrics: float):
        """
        Log metrics to all enabled backends.

        Args:
            step: Training step or epoch
            **metrics: Key-value pairs of metrics to log
        """
        if not metrics:
            return

        # Log to W&B
        if self._wandb_run:
            try:
                self._wandb_run.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metrics to W&B: {e}")

        # Log to TensorBoard
        if self._tb_writer:
            try:
                for key, value in metrics.items():
                    self._tb_writer.add_scalar(key, value, step)
            except Exception as e:
                logger.warning(f"Failed to log metrics to TensorBoard: {e}")

    def log_params(self, **params: Any):
        """
        Log parameters to all enabled backends.

        Args:
            **params: Key-value pairs of parameters to log
        """
        if not params:
            return

        # Log to W&B
        if self._wandb_run:
            try:
                for key, value in params.items():
                    self._wandb_run.config[key] = value
            except Exception as e:
                logger.warning(f"Failed to log parameters to W&B: {e}")

        # Log to TensorBoard
        if self._tb_writer:
            try:
                # TensorBoard doesn't have a direct equivalent for parameters,
                # so we log them as text
                import json

                self._tb_writer.add_text(
                    "parameters", f"```json\n{json.dumps(params, indent=2)}\n```"
                )
            except Exception as e:
                logger.warning(f"Failed to log parameters to TensorBoard: {e}")

    def set_summary(self, **summary: Any):
        """
        Set summary metrics (e.g., final metrics).

        Args:
            **summary: Key-value pairs of summary metrics
        """
        if not summary:
            return

        # Set W&B summary
        if self._wandb_run:
            try:
                for key, value in summary.items():
                    self._wandb_run.summary[key] = value
            except Exception as e:
                logger.warning(f"Failed to set summary in W&B: {e}")

        # TensorBoard doesn't have a direct equivalent for summary

    def log_artifact(
        self, path: Union[str, Path], name: Optional[str] = None, artifact_type: str = "model"
    ):
        """
        Log an artifact (e.g., model checkpoint) to all enabled backends.

        Args:
            path: Path to the artifact file
            name: Optional name for the artifact (defaults to filename)
            artifact_type: Type of artifact (e.g., model, dataset)
        """
        if not self._config.log_artifacts:
            logger.info(f"Artifact logging disabled, skipping: {path}")
            return

        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning(f"Artifact file not found, skipping: {path}")
            return

        artifact_name = name or path_obj.name

        # Log to W&B
        if self._wandb_run:
            try:
                import wandb

                artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
                artifact.add_file(str(path_obj))
                self._wandb_run.log_artifact(artifact)
                logger.info(f"Logged artifact to W&B: {artifact_name}")
            except ImportError:
                logger.warning("W&B not installed, skipping artifact logging")
            except Exception as e:
                logger.warning(f"Failed to log artifact to W&B: {e}")

        # TensorBoard doesn't have a direct equivalent for artifacts

    def close(self):
        """Close all backends and release resources."""
        # Close W&B
        if self._wandb_run:
            try:
                self._wandb_run.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.warning(f"Failed to finish W&B run: {e}")
            self._wandb_run = None

        # Close TensorBoard
        if self._tb_writer:
            try:
                self._tb_writer.close()
                logger.info("TensorBoard writer closed")
            except Exception as e:
                logger.warning(f"Failed to close TensorBoard writer: {e}")
            self._tb_writer = None
