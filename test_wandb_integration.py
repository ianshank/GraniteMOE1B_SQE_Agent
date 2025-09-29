#!/usr/bin/env python3
"""
Test script for W&B integration.

This script demonstrates how to use the Weights & Biases integration
with our telemetry framework for ML experiment tracking.
"""

import os
import sys
import logging
import time
import numpy as np
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our telemetry utilities
try:
    from granite_test_generator.src.config import load_telemetry_from_sources
    from granite_test_generator.src.telemetry import ExperimentLogger
except ImportError:
    # Try local import path
    sys.path.insert(0, str(Path(__file__).resolve().parent / "granite-test-generator" / "src"))
    from config import load_telemetry_from_sources
    from telemetry import ExperimentLogger


class SimpleModel:
    """A simple model that generates random predictions."""
    
    def __init__(self, input_dim=4, output_dim=2):
        """Initialize a simple model with random weights.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def predict(self, x):
        """Make predictions with the model.
        
        Args:
            x: Input array of shape (batch_size, input_dim)
            
        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        return np.dot(x, self.weights) + self.bias
    
    def save(self, path):
        """Save the model weights.
        
        Args:
            path: Path to save the weights
        """
        np.savez(
            path, 
            weights=self.weights, 
            bias=self.bias,
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )
        logger.info(f"Model saved to {path}")


def generate_fake_data(num_samples=100, input_dim=4):
    """Generate fake data for testing.
    
    Args:
        num_samples: Number of samples to generate
        input_dim: Input dimension
        
    Returns:
        Tuple of (inputs, labels)
    """
    X = np.random.randn(num_samples, input_dim)
    y = np.random.randint(0, 2, size=num_samples)
    return X, y


def compute_accuracy(predictions, labels):
    """Compute classification accuracy.
    
    Args:
        predictions: Model predictions
        labels: True labels
        
    Returns:
        Accuracy score
    """
    pred_classes = predictions.argmax(axis=1)
    return np.mean(pred_classes == labels)


def run_fake_training(model, config, experiment_logger):
    """Run a fake training loop with W&B logging.
    
    Args:
        model: Model to train
        config: Configuration dictionary
        experiment_logger: ExperimentLogger instance for telemetry
    """
    input_dim = config.get("input_dim", 4)
    num_epochs = config.get("num_epochs", 5)
    batch_size = config.get("batch_size", 10)
    
    # Generate fake data
    X, y = generate_fake_data(num_samples=100, input_dim=input_dim)
    
    # Log training configuration
    experiment_logger.log_params(
        input_dim=input_dim,
        num_epochs=num_epochs,
        batch_size=batch_size,
        model_type="SimpleModel"
    )
    
    # Mock training loop
    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")
        
        # Mock batch training
        train_losses = []
        accuracies = []
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            # Get predictions
            preds = model.predict(batch_X)
            
            # Calculate mock loss (decreasing over time)
            loss = 1.0 - 0.1 * epoch - 0.01 * (i // batch_size) + 0.05 * np.random.random()
            loss = max(0.1, loss)  # Ensure loss doesn't go below 0.1
            train_losses.append(loss)
            
            # Calculate accuracy
            acc = compute_accuracy(preds, batch_y)
            accuracies.append(acc)
            
            # Log metrics for this step
            step = (epoch - 1) * (len(X) // batch_size) + (i // batch_size)
            experiment_logger.log_metrics(
                step,
                loss=loss,
                accuracy=acc,
                epoch=epoch
            )
            
            # Simulate training time
            time.sleep(0.1)
        
        # Log epoch metrics
        avg_loss = np.mean(train_losses)
        avg_acc = np.mean(accuracies)
        logger.info(f"Epoch {epoch} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        experiment_logger.log_metrics(
            epoch,
            epoch_loss=avg_loss,
            epoch_accuracy=avg_acc,
            epoch=epoch
        )
    
    # Save model checkpoint
    model_path = Path("artifacts") / "model"
    model_path.parent.mkdir(exist_ok=True, parents=True)
    model.save(model_path.with_suffix(".npz"))
    
    # Log model as artifact
    experiment_logger.log_artifact(
        model_path.with_suffix(".npz"),
        name="model",
        type="model"
    )
    
    # Set final metrics in summary
    experiment_logger.set_summary(
        final_loss=avg_loss,
        final_accuracy=avg_acc,
        epochs_completed=num_epochs
    )


def main():
    """Main function to run the W&B integration test."""
    logger.info("Starting W&B integration test")
    
    # Create experiment configuration
    config = {
        "input_dim": 4,
        "num_epochs": 5,
        "batch_size": 10,
        "learning_rate": 0.01,
        "model": {
            "type": "simple",
            "layers": [4, 8, 2]
        }
    }
    
    # Load telemetry configuration from environment
    telemetry_cfg = load_telemetry_from_sources()
    
    # Create model
    model = SimpleModel(input_dim=config["input_dim"], output_dim=2)
    
    # Start experiment logging
    with ExperimentLogger(telemetry_cfg, config) as experiment:
        # Run fake training
        run_fake_training(model, config, experiment)
    
    logger.info("W&B integration test completed")
    logger.info("Check your W&B dashboard or TensorBoard for results")
    
    # If using offline mode, print instructions for syncing
    if os.environ.get("WANDB_MODE") == "offline":
        logger.info(
            "Using W&B offline mode. To sync runs, use:\n"
            "wandb sync wandb/offline-run-*"
        )


if __name__ == "__main__":
    main()
