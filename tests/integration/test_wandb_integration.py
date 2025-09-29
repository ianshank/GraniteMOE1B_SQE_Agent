#!/usr/bin/env python3
"""
Integration tests for W&B API interactions.

These tests verify the proper integration with the W&B API
for experiment tracking and artifact management.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, List

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if W&B is available
try:
    import wandb
    HAS_WANDB = True
    logger.info("W&B module available for integration testing")
except ImportError:
    HAS_WANDB = False
    logger.warning("W&B module not available; integration tests will be skipped")

# Check if API key is available
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
HAS_API_KEY = bool(WANDB_API_KEY)
if not HAS_API_KEY:
    logger.warning("WANDB_API_KEY not set; tests requiring API access will be skipped")

from src.config import TelemetryConfig
from src.telemetry import ExperimentLogger


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, name: str = "mock_model"):
        """Initialize the mock model."""
        self.name = name
        self.params = {
            "layers": 2,
            "hidden_size": 128,
            "dropout": 0.1,
        }
    
    def __call__(self, inputs: Any) -> Any:
        """Forward pass."""
        # Simple mock output
        if isinstance(inputs, (list, tuple)):
            return [i * 2 for i in inputs]
        return inputs * 2


class TestWandBIntegration:
    """Integration tests for W&B."""
    
    @pytest.mark.skipif(not HAS_WANDB, reason="W&B not available")
    def test_offline_run_creation(self, tmp_path: Path) -> None:
        """Test creating an offline W&B run."""
        # Set up offline mode
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = str(tmp_path)
        
        # Configure telemetry
        config = TelemetryConfig(
            enable_wandb=True,
            wandb_project="test-project",
            wandb_entity="test-entity",
            wandb_run_name="test-offline-run",
            wandb_tags=["test", "offline"],
        )
        
        # Create experiment logger
        with ExperimentLogger(config, {"model": "test_model"}) as exp_logger:
            # Log metrics
            exp_logger.log_metrics(1, loss=0.5, accuracy=0.8)
            exp_logger.log_params(learning_rate=0.01, batch_size=32)
            
            # Create and log artifact
            artifact_path = tmp_path / "model.json"
            with open(artifact_path, "w") as f:
                json.dump({"weights": [0.1, 0.2, 0.3]}, f)
            exp_logger.log_artifact(artifact_path, name="model-weights", type="model")
            
            # Set summary
            exp_logger.set_summary(final_loss=0.3, final_accuracy=0.9)
        
        # Check that offline run was created
        run_dir = list(tmp_path.glob("wandb/offline-run-*"))
        assert len(run_dir) > 0, "No offline run directory created"
        
        # Check that files were created
        run_files = list(run_dir[0].glob("*"))
        file_names = [f.name for f in run_files]
        logger.debug(f"Offline run files: {file_names}")
        
        # Different W&B versions may have different file structures
        # Just check that some expected files/directories exist
        assert any(f.endswith(".wandb") for f in file_names) or "files" in file_names, "No .wandb file or files directory created"
        
        # Clean up
        os.environ.pop("WANDB_MODE", None)
        os.environ.pop("WANDB_DIR", None)
    
    @pytest.mark.skipif(not (HAS_WANDB and HAS_API_KEY), reason="W&B or API key not available")
    def test_real_api_interaction(self, tmp_path: Path) -> None:
        """Test interaction with the real W&B API."""
        # Configure telemetry with real API access
        config = TelemetryConfig(
            enable_wandb=True,
            wandb_project=os.environ.get("WANDB_PROJECT", "telemetry-test"),
            wandb_entity=os.environ.get("WANDB_ENTITY", None),
            wandb_run_name=f"api-test-{os.getpid()}",  # Unique run name
            wandb_tags=["test", "api"],
        )
        
        # Create experiment logger
        with ExperimentLogger(config, {"model": "api_test_model"}) as exp_logger:
            # Log metrics over multiple steps
            for step in range(1, 6):
                exp_logger.log_metrics(
                    step,
                    loss=1.0 - step * 0.15,
                    accuracy=0.5 + step * 0.08,
                )
            
            # Log parameters
            exp_logger.log_params(
                learning_rate=0.01,
                batch_size=32,
                optimizer="adam",
                epochs=5,
            )
            
            # Create and log artifact
            artifact_path = tmp_path / "model.json"
            with open(artifact_path, "w") as f:
                json.dump({"weights": [0.1, 0.2, 0.3]}, f)
            exp_logger.log_artifact(artifact_path, name="model-weights", type="model")
            
            # Set summary
            exp_logger.set_summary(
                final_loss=0.25,
                final_accuracy=0.92,
                training_time=1.23,
            )
        
        # Verify run was created by querying the API
        api = wandb.Api()
        entity = os.environ.get("WANDB_ENTITY")
        project = os.environ.get("WANDB_PROJECT", "telemetry-test")
        
        # List runs in the project
        runs = api.runs(f"{entity}/{project}" if entity else project)
        run_names = [run.name for run in runs]
        logger.debug(f"Project runs: {run_names}")
        
        # Find our run
        our_run_name = f"api-test-{os.getpid()}"
        assert any(our_run_name in name for name in run_names), f"Run {our_run_name} not found in project"
    
    @pytest.mark.skipif(not HAS_WANDB, reason="W&B not available")
    def test_tensorboard_and_wandb_integration(self, tmp_path: Path) -> None:
        """Test using both TensorBoard and W&B together."""
        # Set up offline mode for W&B
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = str(tmp_path)
        
        # Configure telemetry with both enabled
        config = TelemetryConfig(
            enable_wandb=True,
            wandb_project="test-project",
            wandb_run_name="test-both-run",
            enable_tensorboard=True,
            tb_log_dir=str(tmp_path / "tb_logs"),
        )
        
        # Create experiment logger
        with ExperimentLogger(config, {"model": "dual_test_model"}) as exp_logger:
            # Log metrics
            for step in range(1, 11):
                exp_logger.log_metrics(
                    step,
                    loss=1.0 - step * 0.08,
                    accuracy=0.5 + step * 0.05,
                )
        
        # Check that both W&B and TensorBoard files were created
        wandb_dir = list(tmp_path.glob("wandb/offline-run-*"))
        assert len(wandb_dir) > 0, "No W&B offline run directory created"
        
        tb_dir = tmp_path / "tb_logs"
        assert tb_dir.exists(), "TensorBoard log directory not created"
        assert any(tb_dir.glob("*")), "No TensorBoard files created"
        
        # Clean up
        os.environ.pop("WANDB_MODE", None)
        os.environ.pop("WANDB_DIR", None)
    
    @pytest.mark.skipif(not HAS_WANDB, reason="W&B not available")
    def test_full_evaluation_workflow(self, tmp_path: Path) -> None:
        """Test a full evaluation workflow with model and metrics."""
        # Set up offline mode
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = str(tmp_path)
        
        # Configure telemetry
        config = TelemetryConfig(
            enable_wandb=True,
            wandb_project="test-project",
            wandb_run_name="test-workflow-run",
        )
        
        # Create mock model and data
        model = MockModel()
        
        # Simple dataloader with 5 samples
        class SimpleDataLoader:
            def __init__(self):
                self.data = [
                    ([1.0], 2.0),    # Correct
                    ([2.0], 3.0),    # Off by 1
                    ([3.0], 6.0),    # Correct
                    ([4.0], 7.0),    # Off by 1
                    ([5.0], 10.0),   # Correct
                ]
            
            def __iter__(self):
                return iter(self.data)
        
        dataloader = SimpleDataLoader()
        
        # Create experiment logger
        with ExperimentLogger(config, {"model": model.name, **model.params}) as exp_logger:
            # Import here to avoid circular import
            from src.eval.evaluate import evaluate
            
            # Run evaluation
            metrics = evaluate(
                model,
                dataloader,
                task_type="regression",
                experiment_logger=exp_logger,
                output_dir=tmp_path / "eval_results",
                epoch=1,
            )
            
            # Log additional metrics
            exp_logger.log_metrics(2, training_loss=0.2, validation_loss=0.3)
            
            # Set summary
            exp_logger.set_summary(**metrics)
        
        # Check that evaluation report was created
        eval_dir = tmp_path / "eval_results"
        assert eval_dir.exists(), "Evaluation directory not created"
        
        report_path = eval_dir / "eval_report.json"
        assert report_path.exists(), "Evaluation report not created"
        
        # Check report contents
        with open(report_path) as f:
            report = json.load(f)
        
        assert "mae" in report, "MAE not in evaluation report"
        assert "rmse" in report, "RMSE not in evaluation report"
        assert "r2" in report, "R² not in evaluation report"


if __name__ == "__main__":
    # Run the tests directly if file is executed
    pytest.main(["-v", __file__])