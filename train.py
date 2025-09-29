#!/usr/bin/env python3
"""
Training harness for the Granite MoE Test Generator.

This script provides a simple training loop with telemetry integration.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Robust imports without hardcoded path assumptions
# 1) Try importing directly (when PYTHONPATH is set appropriately)
# 2) If that fails, attempt to discover likely local paths and retry
# 3) If still unavailable, continue in minimal mode with guidance
try:
    from src.config import load_telemetry_from_sources
    from src.eval.evaluate import evaluate
except ImportError:
    # Attempt local discovery paths
    project_dir = Path(__file__).resolve().parent
    candidates = [
        project_dir / "granite-test-generator" / "src",
        project_dir / "granite-test-generator",
    ]

    added_any = False
    for candidate in candidates:
        if candidate.exists():
            path_str = str(candidate)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
                added_any = True

    try:
        from src.config import load_telemetry_from_sources  # type: ignore
        from src.eval.evaluate import evaluate  # type: ignore
        if added_any:
            logger.debug("Loaded telemetry modules after augmenting sys.path.")
    except ImportError:
        logger.warning(
            "Telemetry modules not available. Running in minimal mode.\n"
            "To enable telemetry/evaluation imports, either set PYTHONPATH to include\n"
            "granite-test-generator/src or install the package in editable mode:\n"
            "    pip install -e granite-test-generator/src"
        )
        load_telemetry_from_sources = None  # type: ignore
        evaluate = None  # type: ignore


def run_training(args_list: Optional[List[str]] = None) -> Dict[str, Union[float, str]]:
    """
    Run a training loop with telemetry integration.
    
    Args:
        args_list: Optional list of command line arguments
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Training harness")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--task-type", type=str, default="classification", help="Task type")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    
    # Telemetry arguments
    parser.add_argument("--enable-wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, help="W&B project name")
    parser.add_argument("--enable-tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--tb-log-dir", type=str, help="TensorBoard log directory")
    parser.add_argument("--log-checkpoints", action="store_true", help="Log checkpoints as artifacts")
    
    args = parser.parse_args(args_list)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    
    # Load telemetry configuration if available
    telemetry = None
    if load_telemetry_from_sources:
        telemetry = load_telemetry_from_sources(
            cli_args=args,
            config_snapshot={
                "model": {"type": "test-model"},
                "data": {"type": "synthetic"}
            }
        )
    
    # Simulate training loop
    logger.info(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    
    # Create a dummy model checkpoint
    checkpoint_path = checkpoint_dir / "model.pt"
    with open(checkpoint_path, "w") as f:
        f.write("dummy checkpoint")
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Generate dummy evaluation metrics
    metrics = {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.80,
        "f1": 0.81,
        "task_type": args.task_type
    }
    
    # Save evaluation report
    report_path = eval_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved evaluation report to {report_path}")
    
    # Run evaluation with telemetry if available
    if evaluate and telemetry:
        try:
            evaluate(
                model=None,  # No actual model in this stub
                task_type=args.task_type,
                predictions=[0, 1, 1, 0],
                targets=[0, 1, 0, 0],
                telemetry=telemetry,
                checkpoint_path=str(checkpoint_path) if args.log_checkpoints else None,
            )
        except Exception as e:
            # Use the module logger, not the ExperimentLogger
            logger.warning(f"Evaluation with telemetry failed: {e}")
    
    return metrics


if __name__ == "__main__":
    run_training()
