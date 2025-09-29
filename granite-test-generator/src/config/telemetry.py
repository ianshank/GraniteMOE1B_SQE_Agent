"""Telemetry configuration models and helpers."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from pydantic import BaseModel, Field, validator

LOGGER = logging.getLogger(__name__)

_BOOL_TRUTHY = {"1", "true", "t", "yes", "y", "on"}
_BOOL_FALSY = {"0", "false", "f", "no", "n", "off"}


def _normalize_bool(value: Optional[Any]) -> Optional[bool]:
    """Convert supported truthy/falsy representations into booleans."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _BOOL_TRUTHY:
            return True
        if lowered in _BOOL_FALSY:
            return False
        LOGGER.debug("Unable to coerce boolean from string '%s'", value)
        return None
    LOGGER.debug("Unable to coerce boolean from type %s", type(value).__name__)
    return None


def _split_tags(raw_tags: Optional[Any]) -> List[str]:
    """
    Return a list of telemetry tags.
    
    Args:
        raw_tags: Raw tag input which can be a string, list, or other value
        
    Returns:
        List of tag strings
        
    This function handles various input formats:
    - Comma-separated strings: "tag1,tag2,tag3" -> ["tag1", "tag2", "tag3"]
    - Lists of strings: ["tag1", "tag2"] -> ["tag1", "tag2"]
    - Lists with comma-separated items: ["tag1,tag2", "tag3"] -> ["tag1", "tag2", "tag3"]
    - Non-string/list values are converted to string: 123 -> ["123"]
    """
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
    if isinstance(raw_tags, Iterable) and not isinstance(raw_tags, (str, bytes)):
        tags: List[str] = []
        for item in raw_tags:
            if not item:
                continue
            tags.extend(_split_tags(str(item)))
        return tags
    # Handle non-iterable values (like integers) by converting to string
    if raw_tags:
        return [str(raw_tags)]
    return []


class TelemetryConfig(BaseModel):
    """Declarative telemetry configuration for experiment tracking."""

    # W&B settings
    enable_wandb: bool = Field(default=False)
    wandb_project: Optional[str] = Field(default=None)
    wandb_entity: Optional[str] = Field(default=None)
    wandb_run_name: Optional[str] = Field(default=None)
    wandb_tags: List[str] = Field(default_factory=list)
    wandb_group: Optional[str] = Field(default=None)
    wandb_job_type: Optional[str] = Field(default=None)
    wandb_name: Optional[str] = Field(default=None)
    wandb_notes: Optional[str] = Field(default=None)
    
    # TensorBoard settings
    enable_tensorboard: bool = Field(default=False)
    tb_log_dir: str = Field(default="runs/")
    
    # General settings
    log_interval_steps: int = Field(default=50, ge=1)
    log_artifacts: bool = Field(default=True)

    @validator("tb_log_dir")
    def _validate_tb_log_dir(cls, value: str) -> str:
        if not value:
            raise ValueError("TensorBoard log directory must not be empty")
        return value

    @validator("wandb_tags", pre=True)
    def _ensure_tags_list(cls, value: Any) -> List[str]:
        return _split_tags(value)

    def merged_with(self, **overrides: Any) -> "TelemetryConfig":
        """
        Create a new config with the provided overrides applied.
        
        Args:
            **overrides: Key-value pairs to override in the config
            
        Returns:
            A new TelemetryConfig instance with overrides applied
        """
        data = self.model_dump()
        data.update({k: v for k, v in overrides.items() if v is not None})
        
        # Handle validation errors for specific fields
        if "log_interval_steps" in overrides:
            try:
                value = int(overrides["log_interval_steps"])
                if value < 1:
                    # Use default value instead of invalid value
                    data["log_interval_steps"] = 50
            except (ValueError, TypeError):
                # Use default value for invalid values
                data["log_interval_steps"] = 50
                
        return TelemetryConfig(**data)


def load_telemetry_from_sources(
    cli_args: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
) -> TelemetryConfig:
    """Materialise ``TelemetryConfig`` from CLI arguments and environment variables."""
    env = dict(env or os.environ)

    env_enable_wandb = _normalize_bool(env.get("ENABLE_WANDB"))
    env_enable_tb = _normalize_bool(env.get("ENABLE_TENSORBOARD"))

    env_tags = env.get("WANDB_TAGS")

    base_cfg = TelemetryConfig(
        enable_wandb=env_enable_wandb or False,
        wandb_project=env.get("WANDB_PROJECT"),
        wandb_entity=env.get("WANDB_ENTITY"),
        wandb_run_name=env.get("WANDB_RUN_NAME"),
        wandb_tags=_split_tags(env_tags),
        wandb_group=env.get("WANDB_GROUP"),
        wandb_job_type=env.get("WANDB_JOB_TYPE"),
        wandb_name=env.get("WANDB_NAME"),
        wandb_notes=env.get("WANDB_NOTES"),
        enable_tensorboard=env_enable_tb or False,
        tb_log_dir=env.get("TB_LOG_DIR", "runs/"),
        log_interval_steps=int(env.get("LOG_INTERVAL_STEPS", 50)),
        log_artifacts=env.get("LOG_ARTIFACTS", "").lower() not in ("0", "false", "no"),
    )

    if not cli_args:
        return base_cfg

    def _get(mapping: Mapping[str, Any], key: str) -> Any:
        if hasattr(mapping, key):
            return getattr(mapping, key)
        return mapping.get(key)

    cli_enable_wandb = _normalize_bool(_get(cli_args, "enable_wandb"))
    cli_enable_tb = _normalize_bool(_get(cli_args, "enable_tensorboard"))

    overrides: MutableMapping[str, Any] = {}
    if cli_enable_wandb is not None:
        overrides["enable_wandb"] = cli_enable_wandb
    if cli_enable_tb is not None:
        overrides["enable_tensorboard"] = cli_enable_tb

    for key in (
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
        "wandb_tags",
        "wandb_group",
        "wandb_job_type",
        "wandb_name",
        "wandb_notes",
        "tb_log_dir",
        "log_interval_steps",
        "log_artifacts",
    ):
        value = _get(cli_args, key)
        if value is not None:
            overrides[key] = value

    if "wandb_tags" in overrides:
        overrides["wandb_tags"] = _split_tags(overrides["wandb_tags"])

    if "log_interval_steps" in overrides:
        try:
            overrides["log_interval_steps"] = int(overrides["log_interval_steps"])
            if overrides["log_interval_steps"] < 1:
                LOGGER.warning("Invalid log interval: %s (must be >= 1)", overrides["log_interval_steps"])
                overrides.pop("log_interval_steps", None)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid log interval override %s", overrides["log_interval_steps"])
            overrides.pop("log_interval_steps", None)

    result = base_cfg.merged_with(**overrides)
    
    # Add config snapshot to notes if provided
    if config_snapshot and result.enable_wandb:
        import json

        notes = result.wandb_notes or ""
        if notes:
            notes += "\n\n"
        notes += f"Config: {json.dumps(config_snapshot, indent=2)}"
        result.wandb_notes = notes

    return result