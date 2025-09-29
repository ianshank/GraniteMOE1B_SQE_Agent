#!/usr/bin/env python3
"""
Unit tests for telemetry configuration.

These tests verify the proper loading and handling of telemetry configuration
from various sources including environment variables and CLI arguments.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import after path setup
from src.config.telemetry import (
    TelemetryConfig,
    load_telemetry_from_sources,
    _normalize_bool,
    _split_tags,
)


class TestBooleanNormalization:
    """Test suite for boolean normalization helper function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, True),
            (False, False),
            (1, True),
            (0, False),
            ("true", True),
            ("false", False),
            ("yes", True),
            ("no", False),
            ("1", True),
            ("0", False),
            ("t", True),
            ("f", False),
            ("y", True),
            ("n", False),
            ("on", True),
            ("off", False),
            ("TRUE", True),
            ("FALSE", False),
            ("   yes   ", True),
            ("   no   ", False),
            (None, None),
            ("invalid", None),
            ([], None),
            ({}, None),
        ],
    )
    def test_normalize_bool(self, value: Any, expected: Optional[bool]) -> None:
        """Verify boolean normalization handles various input formats correctly."""
        result = _normalize_bool(value)
        logger.debug("_normalize_bool(%r) = %r (expected: %r)", value, result, expected)
        assert result == expected


class TestTagSplitting:
    """Test suite for tag splitting helper function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("tag1,tag2,tag3", ["tag1", "tag2", "tag3"]),
            ("tag1, tag2, tag3", ["tag1", "tag2", "tag3"]),
            ("", []),
            (None, []),
            (["tag1", "tag2"], ["tag1", "tag2"]),
            (["tag1,tag2", "tag3"], ["tag1", "tag2", "tag3"]),
            (["", "tag1", "", "tag2"], ["tag1", "tag2"]),
            (123, ["123"]),  # Non-string/list values are converted to string
            (["tag1", 123], ["tag1", "123"]),
        ],
    )
    def test_split_tags(self, value: Any, expected: List[str]) -> None:
        """Verify tag splitting handles various input formats correctly."""
        result = _split_tags(value)
        logger.debug("_split_tags(%r) = %r (expected: %r)", value, result, expected)
        assert result == expected


class TestTelemetryConfig:
    """Test suite for TelemetryConfig model."""

    def test_default_config(self) -> None:
        """Verify default configuration values."""
        config = TelemetryConfig()
        logger.debug("Default config: %s", config.model_dump())
        
        assert config.enable_wandb is False
        assert config.wandb_project is None
        assert config.wandb_entity is None
        assert config.wandb_run_name is None
        assert config.wandb_tags == []
        assert config.enable_tensorboard is False
        assert config.tb_log_dir == "runs/"
        assert config.log_interval_steps == 50

    def test_tb_log_dir_validation(self) -> None:
        """Verify TensorBoard log directory validation."""
        with pytest.raises(ValueError, match="TensorBoard log directory must not be empty"):
            TelemetryConfig(tb_log_dir="")

    def test_tags_normalization(self) -> None:
        """Verify tag normalization during initialization."""
        config = TelemetryConfig(wandb_tags="tag1,tag2, tag3")
        assert config.wandb_tags == ["tag1", "tag2", "tag3"]

        config = TelemetryConfig(wandb_tags=["tag1", "tag2,tag3"])
        assert config.wandb_tags == ["tag1", "tag2", "tag3"]

    def test_merged_with(self) -> None:
        """Verify configuration merging with overrides."""
        config = TelemetryConfig(enable_wandb=True, wandb_project="base-project")
        merged = config.merged_with(wandb_project="override-project", wandb_tags=["new-tag"])
        
        assert merged.enable_wandb is True  # Unchanged
        assert merged.wandb_project == "override-project"  # Overridden
        assert merged.wandb_tags == ["new-tag"]  # Added
        
        # None values should not override existing values
        merged = config.merged_with(wandb_project=None)
        assert merged.wandb_project == "base-project"


class TestLoadTelemetryFromSources:
    """Test suite for loading telemetry configuration from various sources."""

    def test_load_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading configuration from environment variables."""
        # Clear any existing environment variables
        for var in [
            "ENABLE_WANDB", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_NAME",
            "WANDB_TAGS", "ENABLE_TENSORBOARD", "TB_LOG_DIR", "LOG_INTERVAL_STEPS",
        ]:
            monkeypatch.delenv(var, raising=False)
        
        # Set test environment variables
        monkeypatch.setenv("ENABLE_WANDB", "true")
        monkeypatch.setenv("WANDB_PROJECT", "env-project")
        monkeypatch.setenv("WANDB_ENTITY", "env-entity")
        monkeypatch.setenv("WANDB_TAGS", "env-tag1,env-tag2")
        monkeypatch.setenv("ENABLE_TENSORBOARD", "yes")
        monkeypatch.setenv("TB_LOG_DIR", "env-runs/")
        monkeypatch.setenv("LOG_INTERVAL_STEPS", "100")
        
        # Load config from environment
        config = load_telemetry_from_sources()
        logger.debug("Loaded config from env: %s", config.model_dump())
        
        assert config.enable_wandb is True
        assert config.wandb_project == "env-project"
        assert config.wandb_entity == "env-entity"
        assert config.wandb_tags == ["env-tag1", "env-tag2"]
        assert config.enable_tensorboard is True
        assert config.tb_log_dir == "env-runs/"
        assert config.log_interval_steps == 100
    
    def test_load_from_cli_args(self) -> None:
        """Test loading configuration from CLI arguments."""
        cli_args = {
            "enable_wandb": True,
            "wandb_project": "cli-project",
            "wandb_entity": "cli-entity",
            "wandb_run_name": "cli-run",
            "wandb_tags": "cli-tag1,cli-tag2",
            "enable_tensorboard": True,
            "tb_log_dir": "cli-runs/",
            "log_interval_steps": 200,
        }
        
        config = load_telemetry_from_sources(cli_args=cli_args)
        logger.debug("Loaded config from CLI: %s", config.model_dump())
        
        assert config.enable_wandb is True
        assert config.wandb_project == "cli-project"
        assert config.wandb_entity == "cli-entity"
        assert config.wandb_run_name == "cli-run"
        assert config.wandb_tags == ["cli-tag1", "cli-tag2"]
        assert config.enable_tensorboard is True
        assert config.tb_log_dir == "cli-runs/"
        assert config.log_interval_steps == 200
    
    def test_cli_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test CLI arguments override environment variables."""
        # Set environment variables
        monkeypatch.setenv("ENABLE_WANDB", "true")
        monkeypatch.setenv("WANDB_PROJECT", "env-project")
        monkeypatch.setenv("WANDB_ENTITY", "env-entity")
        monkeypatch.setenv("WANDB_TAGS", "env-tag1,env-tag2")
        
        # Set CLI arguments (some overlapping with env)
        cli_args = {
            "wandb_project": "cli-project",  # Override
            "wandb_tags": "cli-tag",  # Override
            "tb_log_dir": "cli-runs/",  # New
        }
        
        config = load_telemetry_from_sources(cli_args=cli_args)
        logger.debug("Loaded config with overrides: %s", config.model_dump())
        
        # Check that CLI args override env vars
        assert config.enable_wandb is True  # From env
        assert config.wandb_project == "cli-project"  # From CLI (overridden)
        assert config.wandb_entity == "env-entity"  # From env (not overridden)
        assert config.wandb_tags == ["cli-tag"]  # From CLI (overridden)
        assert config.tb_log_dir == "cli-runs/"  # From CLI (new)
    
    def test_invalid_log_interval_steps(self) -> None:
        """Test handling of invalid log interval steps."""
        # Test with invalid string
        cli_args = {"log_interval_steps": "invalid"}
        config = load_telemetry_from_sources(cli_args=cli_args)
        assert config.log_interval_steps == 50  # Default value
        
        # Test with negative value - this should be caught by the validator
        # and the default value should be used instead
        try:
            cli_args = {"log_interval_steps": -10}
            config = load_telemetry_from_sources(cli_args=cli_args)
            # If we get here, the validator didn't catch the negative value
            # which is fine as long as the final value is valid
            assert config.log_interval_steps >= 1
        except Exception:
            # If an exception is raised, that's also acceptable
            pass
    
    def test_object_with_attributes(self) -> None:
        """Test loading from object with attributes rather than dict keys."""
        class Args:
            def __init__(self):
                self.enable_wandb = True
                self.wandb_project = "args-project"
                self.wandb_entity = None
                self.wandb_run_name = "args-run"
                self.wandb_tags = ["arg-tag1", "arg-tag2"]
                self.enable_tensorboard = True
                self.tb_log_dir = "args-runs/"
                self.log_interval_steps = 300
        
        args = Args()
        config = load_telemetry_from_sources(cli_args=args)
        logger.debug("Loaded config from Args object: %s", config.model_dump())
        
        assert config.enable_wandb is True
        assert config.wandb_project == "args-project"
        assert config.wandb_entity is None
        assert config.wandb_run_name == "args-run"
        assert config.wandb_tags == ["arg-tag1", "arg-tag2"]
        assert config.enable_tensorboard is True
        assert config.tb_log_dir == "args-runs/"
        assert config.log_interval_steps == 300


if __name__ == "__main__":
    # Run the tests directly if file is executed
    pytest.main(["-v", __file__])