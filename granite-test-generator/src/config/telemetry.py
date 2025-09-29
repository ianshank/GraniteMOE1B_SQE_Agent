"""
Telemetry configuration for experiment tracking.

This module provides a configuration model for telemetry and functions to load
configuration from various sources (CLI args, environment variables).
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TelemetryConfig(BaseModel):
    """Configuration for telemetry and experiment tracking."""

    # W&B settings
    enable_wandb: bool = Field(False, description="Enable Weights & Biases logging")
    wandb_project: Optional[str] = Field(None, description="W&B project name")
    wandb_entity: Optional[str] = Field(None, description="W&B entity (username/team)")
    wandb_tags: List[str] = Field(default_factory=list, description="W&B tags for the run")
    wandb_group: Optional[str] = Field(None, description="W&B group for the run")
    wandb_job_type: Optional[str] = Field(None, description="W&B job type")
    wandb_name: Optional[str] = Field(None, description="W&B run name")
    wandb_notes: Optional[str] = Field(None, description="W&B run notes")

    # TensorBoard settings
    enable_tensorboard: bool = Field(False, description="Enable TensorBoard logging")
    tb_log_dir: str = Field("runs", description="TensorBoard log directory")

    # General settings
    log_interval_steps: int = Field(10, description="Log metrics every N steps")
    log_artifacts: bool = Field(True, description="Log artifacts (models, etc.)")

    def merged_with(self, other: "TelemetryConfig") -> "TelemetryConfig":
        """Merge with another config, with other taking precedence."""
        data = self.model_dump()
        other_data = other.model_dump()

        # Only override non-None/non-default values
        for key, value in other_data.items():
            if key == "wandb_tags" and value:
                data[key].extend([t for t in value if t not in data[key]])
            elif key == "log_interval_steps" and value != 10:
                try:
                    data[key] = value
                except Exception:
                    logger.warning(
                        f"Invalid log_interval_steps value: {value}, using default: {data[key]}"
                    )
            elif value not in (None, False, [], "runs"):
                data[key] = value

        return TelemetryConfig(**data)


def _split_tags(tags_str: Optional[Union[str, List[str], int]]) -> List[str]:
    """Split comma-separated tags string into a list."""
    if not tags_str:
        return []
    if isinstance(tags_str, list):
        return tags_str
    if isinstance(tags_str, (int, float)):
        return [str(tags_str)]
    return [t.strip() for t in str(tags_str).split(",") if t.strip()]


def load_telemetry_from_sources(
    env_vars: Optional[Dict[str, str]] = None,
    cli_args: Optional[Any] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
) -> TelemetryConfig:
    """
    Load telemetry configuration from environment variables and CLI arguments.

    Args:
        env_vars: Optional dictionary of environment variables
        cli_args: Optional CLI arguments object with telemetry attributes
        config_snapshot: Optional config snapshot to include in telemetry

    Returns:
        TelemetryConfig instance with merged configuration
    """
    # Start with empty config
    config = TelemetryConfig()

    # Load from environment variables
    env = env_vars or os.environ
    env_config = TelemetryConfig(
        enable_wandb=env.get("ENABLE_WANDB", "").lower() in ("1", "true", "yes"),
        wandb_project=env.get("WANDB_PROJECT"),
        wandb_entity=env.get("WANDB_ENTITY"),
        wandb_tags=_split_tags(env.get("WANDB_TAGS")),
        wandb_group=env.get("WANDB_GROUP"),
        wandb_job_type=env.get("WANDB_JOB_TYPE"),
        wandb_name=env.get("WANDB_NAME"),
        wandb_notes=env.get("WANDB_NOTES"),
        enable_tensorboard=env.get("ENABLE_TENSORBOARD", "").lower()
        in ("1", "true", "yes"),
        tb_log_dir=env.get("TB_LOG_DIR", "runs"),
        log_interval_steps=int(env.get("LOG_INTERVAL_STEPS", "10")),
        log_artifacts=env.get("LOG_ARTIFACTS", "").lower() not in ("0", "false", "no"),
    )
    config = config.merged_with(env_config)

    # Load from CLI arguments if provided
    if cli_args:
        cli_config = TelemetryConfig(
            enable_wandb=getattr(cli_args, "enable_wandb", False),
            wandb_project=getattr(cli_args, "wandb_project", None),
            wandb_entity=getattr(cli_args, "wandb_entity", None),
            wandb_tags=_split_tags(getattr(cli_args, "wandb_tags", None)),
            wandb_group=getattr(cli_args, "wandb_group", None),
            wandb_job_type=getattr(cli_args, "wandb_job_type", None),
            wandb_name=getattr(cli_args, "wandb_name", None),
            wandb_notes=getattr(cli_args, "wandb_notes", None),
            enable_tensorboard=getattr(cli_args, "enable_tensorboard", False),
            tb_log_dir=getattr(cli_args, "tb_log_dir", "runs") or "runs",
            log_interval_steps=getattr(cli_args, "log_interval_steps", 10),
            log_artifacts=getattr(cli_args, "log_artifacts", True),
        )
        config = config.merged_with(cli_config)

    # Add config snapshot to notes if provided
    if config_snapshot and config.enable_wandb:
        import json

        notes = config.wandb_notes or ""
        if notes:
            notes += "\n\n"
        notes += f"Config: {json.dumps(config_snapshot, indent=2)}"
        config.wandb_notes = notes

    return config
