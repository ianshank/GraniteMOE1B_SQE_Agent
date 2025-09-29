#!/usr/bin/env python3
"""
Integration test for the complete telemetry workflow.

This test verifies the entire telemetry and evaluation system works
end-to-end, from configuration to metrics calculation and artifact logging.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "granite-test-generator"))

# Set up logging for test debugging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try importing torch and other optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    logger.info("PyTorch available for end-to-end testing")
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available - skipping model-based integration tests")

from src.config import TelemetryConfig, load_telemetry_from_sources
from src.telemetry import ExperimentLogger
from src.eval.evaluate import evaluate


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestEndToEndWorkflow:
    """Test suite for end-to-end telemetry workflow with a real model."""
    
    def test_train_eval_log_workflow(self, tmp_path: Path) -> None:
        """Test the complete workflow: train, evaluate, and log results."""
        # Create directories
        output_dir = tmp_path / "workflow_test"
        output_dir.mkdir()
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir()
        tb_dir = tmp_path / "tensorboard_logs"
        tb_dir.mkdir()
        
        # Create a simple dataset
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset[:80], batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset[80:], batch_size=16)
        
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 2)
            
            def forward(self, x):
                return self.layer(x)
        
        model = SimpleModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Create telemetry configuration
        config = TelemetryConfig(
            enable_tensorboard=True,
            tb_log_dir=str(tb_dir),
            wandb_tags=["integration-test", "end-to-end"],
        )
        
        # Configure metadata for the run
        metadata = {
            "model": {"type": "SimpleModel", "params": 22},  # 10*2 + 2 = 22 params
            "dataset": {"type": "synthetic", "samples": 100},
            "training": {"epochs": 2, "batch_size": 16, "optimizer": "Adam"}
        }
        
        # Create experiment logger
        with ExperimentLogger(config, metadata) as logger:
            logger.log_params(**metadata)
            
            # Run a simple training loop
            for epoch in range(1, 3):  # 2 epochs
                logger.debug(f"Training epoch {epoch}")
                model.train()
                epoch_loss = 0.0
                correct = 0
                total = 0
                
                for i, (inputs, labels) in enumerate(train_loader):
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    # Track metrics
                    epoch_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    # Log every few steps
                    if i % 2 == 0:
                        step = i + (epoch - 1) * len(train_loader)
                        logger.log_metrics(
                            step,
                            train_loss=loss.item(),
                            train_acc=predicted.eq(labels).sum().item() / labels.size(0)
                        )
                
                # Calculate epoch metrics
                avg_loss = epoch_loss / len(train_loader)
                accuracy = correct / total
                logger.debug(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
                
                # Save checkpoint
                checkpoint_path = checkpoints_dir / f"model_epoch_{epoch}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                logger.log_artifact(checkpoint_path, name=f"checkpoint-epoch-{epoch}", type="model")
                
                # Run evaluation
                eval_metrics = evaluate(
                    model=model,
                    dataloader=val_loader,
                    task_type="classification",
                    experiment_logger=logger,
                    output_dir=output_dir / "eval",
                    epoch=epoch
                )
                logger.debug(f"Evaluation metrics: {eval_metrics}")
                
                # Set summary metrics
                if epoch == 2:  # Final epoch
                    logger.set_summary(
                        final_loss=avg_loss,
                        final_accuracy=accuracy,
                        val_accuracy=eval_metrics.get("accuracy", 0)
                    )
        
        # Verify TensorBoard logs were created
        assert any(tb_dir.glob("*"))
        logger.debug(f"TensorBoard log files: {list(tb_dir.glob('*'))}")
        
        # Verify evaluation report was created
        eval_report = output_dir / "eval" / "eval_report.json"
        assert eval_report.exists()
        
        # Verify checkpoints were created
        assert (checkpoints_dir / "model_epoch_1.pt").exists()
        assert (checkpoints_dir / "model_epoch_2.pt").exists()


class TestTelemetryConfigIntegration:
    """Test suite for telemetry configuration integration."""
    
    def test_config_from_multiple_sources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading configuration from environment, CLI args, and file."""
        # Set environment variables
        monkeypatch.setenv("ENABLE_WANDB", "true")
        monkeypatch.setenv("WANDB_PROJECT", "env-project")
        monkeypatch.setenv("WANDB_TAGS", "env-tag1,env-tag2")
        
        # Create CLI args
        cli_args = {
            "wandb_entity": "cli-entity",
            "tb_log_dir": "cli-logs/",
            "enable_tensorboard": True
        }
        
        # Load config from all sources
        config = load_telemetry_from_sources(cli_args=cli_args)
        
        # Verify combined configuration
        assert config.enable_wandb is True  # From env
        assert config.wandb_project == "env-project"  # From env
        assert config.wandb_entity == "cli-entity"  # From CLI
        assert config.wandb_tags == ["env-tag1", "env-tag2"]  # From env
        assert config.enable_tensorboard is True  # From CLI
        assert config.tb_log_dir == "cli-logs/"  # From CLI
        
        # Create an experiment logger with this config
        with ExperimentLogger(config, {"test": "integration"}) as logger:
            logger.log_metrics(1, test_metric=0.5)
            logger.log_params(source="multiple")
        
        # No asserts needed here - we're testing that it doesn't crash


class TestEvaluationIntegration:
    """Test suite for evaluation integration."""
    
    def test_evaluate_with_filesystem(self, tmp_path: Path) -> None:
        """Test evaluation with real filesystem interactions."""
        output_dir = tmp_path / "eval_test"
        output_dir.mkdir()
        
        # Simple model and dataloader
        model = lambda x: 1  # Always predict class 1
        data = [
            ([0], 1),  # Correct
            ([0], 0),  # Incorrect
            ([0], 1),  # Correct
        ]
        
        # Create telemetry config
        config = TelemetryConfig(
            enable_tensorboard=True,
            tb_log_dir=str(tmp_path / "tb_logs")
        )
        
        # Create experiment logger
        with ExperimentLogger(config, {"model": "test_model"}) as logger:
            # Run evaluation
            metrics = evaluate(
                model=model,
                dataloader=data,
                task_type="classification",
                experiment_logger=logger,
                output_dir=output_dir,
                epoch=1
            )
            
            # Verify metrics
            assert "accuracy" in metrics
            assert metrics["accuracy"] == 2/3  # 2 correct out of 3
            
            # Verify report file was created
            report_path = output_dir / "eval_report.json"
            assert report_path.exists()
            
            with open(report_path) as f:
                report = json.load(f)
            assert "accuracy" in report
            assert report["accuracy"] == 2/3


def test_offline_wandb_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test integration with offline W&B mode."""
    # Set W&B to offline mode
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_PROJECT", "integration-test")
    
    # Set up directories
    output_dir = tmp_path / "wandb_test"
    output_dir.mkdir()
    artifact_path = output_dir / "artifact.txt"
    artifact_path.write_text("Test artifact content")
    
    # Create config with W&B enabled
    config = TelemetryConfig(
        enable_wandb=True,
        wandb_project="integration-test",
        wandb_tags=["offline-test"]
    )
    
    # Create experiment logger
    with ExperimentLogger(config, {"test": "wandb-offline"}) as logger:
        # Log metrics, parameters, and artifact
        for step in range(1, 6):
            logger.log_metrics(
                step,
                metric1=step * 0.1,
                metric2=1.0 - step * 0.1
            )
        
        logger.log_params(
            test_param="value",
            numeric_param=123
        )
        
        logger.log_artifact(artifact_path, name="test-artifact")
        logger.set_summary(final_metric=0.5)
    
    # Check if offline run was created in wandb directory
    wandb_dir = Path("wandb")
    if wandb_dir.exists():
        offline_runs = list(wandb_dir.glob("offline-run-*"))
        logger.info(f"Found offline W&B runs: {offline_runs}")
    
    # No assertion needed - we're testing that the integration doesn't crash


if __name__ == "__main__":
    # Run the tests directly if file is executed
    pytest.main(["-v", __file__])
