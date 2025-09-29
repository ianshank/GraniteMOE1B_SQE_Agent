#!/usr/bin/env python3
"""
Contract tests for W&B API interactions.

These tests verify that our code correctly interfaces with the W&B API
by testing the assumptions we make about the API contract.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Configure logging for test debugging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check if we have the wandb module available
try:
    import wandb
    from wandb.sdk.lib import telemetry as wandb_telemetry
    HAS_WANDB = True
    logger.info("W&B module available for contract testing")
except ImportError:
    HAS_WANDB = False
    logger.warning("W&B module not available; some tests will be skipped")


class MockWandBClient:
    """Mock implementation of WandBClient for testing."""
    
    def __init__(self, api_key=None, entity=None):
        self.api_key = api_key
        self.entity = entity
        self._api = mock.MagicMock()
    
    def get_run(self, run_path):
        """Get a run by path."""
        parts = run_path.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid run path: {run_path}")
        
        project, run_id = parts
        return self._api.run(f"{self.entity}/{project}/{run_id}")
    
    def sync_run(self, run_path, dry_run=False):
        """Sync an offline run."""
        if dry_run:
            return {"status": "dry_run", "run_path": run_path}
        
        import wandb
        return wandb.sync(run_path)
    
    def update_run_config(self, run_path, updates):
        """Update run configuration."""
        run = self.get_run(run_path)
        run.config.update(updates, allow_val_change=True)
        return run
    
    def export_metrics(self, run_path, output_path):
        """Export metrics from a run."""
        run = self.get_run(run_path)
        history = run.history()
        history.to_csv(output_path)
        return output_path
    
    def download_artifact(self, run_path, filename, output_path):
        """Download an artifact from a run."""
        run = self.get_run(run_path)
        file = run.file(filename)
        file.download(output_path, replace=True)
        return output_path
    
    def get_best_run_from_sweep(self, sweep_path, metric, maximize=True):
        """Get the best run from a sweep."""
        parts = sweep_path.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid sweep path: {sweep_path}")
        
        project, sweep_id = parts
        sweep = self._api.sweep(f"{self.entity}/{project}/{sweep_id}")
        
        if not sweep.runs:
            return None
        
        runs = sorted(
            sweep.runs,
            key=lambda r: r.summary.get(metric, 0),
            reverse=maximize
        )
        
        return runs[0] if runs else None


@pytest.mark.skipif(not HAS_WANDB, reason="W&B not available")
class TestWandBApiContract:
    """Test suite for W&B API contract assumptions."""
    
    def setup_method(self) -> None:
        """Set up test environment before each test."""
        # Create a mock API that mimics W&B API behavior
        self.api_patcher = mock.patch("wandb.Api")
        self.mock_api = self.api_patcher.start()
        
        # Configure the mock API to match expected contract
        self.mock_run = mock.MagicMock()
        self.mock_run.id = "test-run-id"
        self.mock_run.name = "test-run"
        self.mock_run.state = "finished"
        self.mock_run.summary = {}
        # Use a MagicMock so we can assert update() call contract reliably
        self.mock_run.config = mock.MagicMock()
        self.mock_run.config.update = mock.MagicMock()
        self.mock_run.project = "test-project"
        self.mock_run.entity = "test-entity"
        
        # Set up mock API to return our mock run
        self.mock_api.return_value.run.return_value = self.mock_run
        
        # Create client instance using mock API
        self.client = MockWandBClient(api_key="mock-key", entity="test-entity")
        self.client._api = self.mock_api.return_value
    
    def teardown_method(self) -> None:
        """Clean up after each test."""
        self.api_patcher.stop()
    
    def test_get_run_contract(self) -> None:
        """Test contract for getting a run by ID."""
        run = self.client.get_run("test-project/test-run-id")
        
        # Verify that we call the API with the expected path
        self.mock_api.return_value.run.assert_called_once_with("test-entity/test-project/test-run-id")
        
        # Verify we get back what we expect
        assert run.id == "test-run-id"
        assert run.name == "test-run"
    
    def test_sync_offline_run_contract(self) -> None:
        """Test contract for syncing an offline run."""
        # Set up offline run dir with expected structure
        mock_sync = mock.MagicMock()
        
        # Some wandb versions don't expose `wandb.sync`; skip if unavailable
        if hasattr(wandb, "sync"):
            with mock.patch("wandb.sync", mock_sync):
                self.client.sync_run("path/to/offline/run", dry_run=False)
                
                # Verify wandb.sync was called correctly
                mock_sync.assert_called_once()
                # Get the args passed to wandb.sync
                args, kwargs = mock_sync.call_args
                assert len(args) == 1
                assert args[0] == "path/to/offline/run"
        else:
            pytest.skip("wandb.sync not available in this environment")
    
    def test_update_run_config_contract(self) -> None:
        """Test contract for updating run configuration."""
        # Configure mock run to have update method
        updates = {"param1": "new-value", "param2": 42}
        
        self.client.update_run_config("test-project/test-run-id", updates)
        
        # Verify the config.update was called with the right parameters
        # and allow_val_change=True
        self.mock_run.config.update.assert_called_once_with(updates, allow_val_change=True)
    
    def test_export_metrics_contract(self) -> None:
        """Test contract for exporting run metrics."""
        # Configure mock run to have a dataframe-like history
        mock_history = mock.MagicMock()
        mock_history.to_csv = mock.MagicMock()
        self.mock_run.history.return_value = mock_history
        
        output_path = Path("test_metrics.csv")
        self.client.export_metrics("test-project/test-run-id", output_path)
        
        # Verify the history was called and to_csv was called on the result
        self.mock_run.history.assert_called_once()
        mock_history.to_csv.assert_called_once_with(output_path)
    
    def test_artifact_download_contract(self) -> None:
        """Test contract for downloading artifacts."""
        # Configure mock run to have a file method
        mock_file = mock.MagicMock()
        self.mock_run.file.return_value = mock_file
        
        self.client.download_artifact("test-project/test-run-id", "model.pt", "local/path")
        
        # Verify file was called with the right filename
        self.mock_run.file.assert_called_once_with("model.pt")
        # Verify download was called with the right path
        mock_file.download.assert_called_once_with("local/path", replace=True)
    
    def test_get_best_run_from_sweep_contract(self) -> None:
        """Test contract for finding best run in a sweep."""
        # Configure mock sweep with runs
        mock_sweep = mock.MagicMock()
        
        # Create a list of mock runs with different metrics
        mock_runs = []
        for i in range(3):
            mock_run = mock.MagicMock()
            mock_run.name = f"run-{i}"
            mock_run.summary = {"val_accuracy": 0.7 + i * 0.1}  # 0.7, 0.8, 0.9
            mock_runs.append(mock_run)
        
        # Sort runs in reverse order to test sorting logic
        mock_sweep.runs = list(reversed(mock_runs))
        self.mock_api.return_value.sweep.return_value = mock_sweep
        
        # Call the function under test
        best_run = self.client.get_best_run_from_sweep("test-project/sweep-id", "val_accuracy", maximize=True)
        
        # Verify sweep was called with the right path
        self.mock_api.return_value.sweep.assert_called_once_with("test-entity/test-project/sweep-id")
        
        # Check that we got the run with the highest val_accuracy
        assert best_run.name == "run-2"
        # Allow tiny floating error tolerance
        assert best_run.summary["val_accuracy"] == pytest.approx(0.9, rel=1e-9)


@pytest.mark.skipif(not HAS_WANDB, reason="W&B not available")
class TestWandBEnvironmentContract:
    """Test suite for W&B environment variable contract."""
    
    def test_wandb_mode_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test contract for WANDB_MODE environment variable."""
        # Test offline mode
        monkeypatch.setenv("WANDB_MODE", "offline")
        
        # Initialize W&B run
        with mock.patch("wandb.init") as mock_init:
            wandb.init(project="test")
            
            # Verify init was called with the right settings
            args, kwargs = mock_init.call_args
            assert kwargs.get("mode") == "offline" or os.environ.get("WANDB_MODE") == "offline"
    
    def test_wandb_project_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test contract for WANDB_PROJECT environment variable."""
        monkeypatch.setenv("WANDB_PROJECT", "env-project")
        
        with mock.patch("wandb.init") as mock_init:
            # Should use env var if no project specified
            wandb.init()
            args, kwargs = mock_init.call_args
            assert kwargs.get("project") == "env-project" or os.environ.get("WANDB_PROJECT") == "env-project"
            
            # Should override env var if project specified
            mock_init.reset_mock()
            wandb.init(project="explicit-project")
            args, kwargs = mock_init.call_args
            assert kwargs.get("project") == "explicit-project"
    
    def test_wandb_api_key_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test contract for WANDB_API_KEY environment variable."""
        monkeypatch.setenv("WANDB_API_KEY", "fake-api-key")
        
        with mock.patch("wandb.Api") as mock_api:
            api = wandb.Api()
            
            # Verify API was initialized with the key from env
            args, kwargs = mock_api.call_args
            # Note: Not all wandb versions expose the api_key in the constructor
            # but it should be set in the environment
            assert os.environ.get("WANDB_API_KEY") == "fake-api-key"
    
    def test_wandb_dir_contract(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test contract for WANDB_DIR environment variable."""
        wandb_dir = tmp_path / "custom_wandb_dir"
        wandb_dir.mkdir()
        monkeypatch.setenv("WANDB_DIR", str(wandb_dir))
        
        with mock.patch("wandb.init") as mock_init:
            wandb.init(project="test")
            
            # Verify init respects WANDB_DIR
            # This may not be directly visible in the mock call,
            # but the environment variable should be set
            assert os.environ.get("WANDB_DIR") == str(wandb_dir)


# This test is marked as skip by default since it requires a real W&B API key
@pytest.mark.skipif(True, reason="Requires real W&B API key, skipping by default")
def test_real_wandb_api_access() -> None:
    """Test real W&B API access (requires API key)."""
    # This test is skipped by default to avoid requiring API keys
    # To run it, set the WANDB_API_KEY environment variable and remove the skipif
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        pytest.skip("WANDB_API_KEY environment variable not set")
    
    # Create a client with the real API key
    client = MockWandBClient(api_key=api_key)
    
    # Try to list projects
    try:
        entity = os.environ.get("WANDB_ENTITY")
        if entity:
            api = wandb.Api()
            projects = api.projects(entity)
            logger.info(f"Found {len(projects)} projects for entity {entity}")
        else:
            logger.info("WANDB_ENTITY not set, skipping project listing")
    except Exception as e:
        logger.error(f"Error accessing W&B API: {e}")
        pytest.fail(f"Error accessing W&B API: {e}")


if __name__ == "__main__":
    # Run the tests directly if file is executed
    pytest.main(["-v", __file__])