# Weights & Biases Integration Tutorial

This tutorial demonstrates how to use Weights & Biases (W&B) with the Granite SQE Agent project for experiment tracking, visualization, and model evaluation.

## 1. Setup and Configuration

First, let's set up our environment and configure W&B:

```python
# Import necessary libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to Python path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

# Import project modules
try:
    from granite_test_generator.src.config import load_telemetry_from_sources
    from granite_test_generator.src.telemetry import ExperimentLogger
except ImportError:
    # Try local import path
    sys.path.insert(0, str(project_root / "granite-test-generator" / "src"))
    from config import load_telemetry_from_sources
    from telemetry import ExperimentLogger
```

### 1.1 Configure W&B Settings

You can configure W&B settings using environment variables. For this tutorial, we'll set them directly in the notebook:

```python
# Set W&B environment variables
os.environ["ENABLE_WANDB"] = "true"
os.environ["WANDB_PROJECT"] = "granite-tutorial"
os.environ["WANDB_RUN_NAME"] = "notebook-demo"
os.environ["WANDB_TAGS"] = "tutorial,notebook,demo"

# Use offline mode for the tutorial
# When you're ready to sync to W&B, set WANDB_MODE to "online"
os.environ["WANDB_MODE"] = "offline"

# If you have a W&B API key, you can set it here
# os.environ["WANDB_API_KEY"] = "your-api-key"  # Uncomment and set your API key
```

## 2. Creating a Simple Model

For this tutorial, we'll create a simple linear model to demonstrate W&B integration:

```python
class SimpleLinearModel:
    """A simple linear model for demonstration purposes."""
    
    def __init__(self, input_dim=2, learning_rate=0.01):
        """Initialize the model with random weights.
        
        Args:
            input_dim: Input dimension
            learning_rate: Learning rate for gradient descent
        """
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn()
        self.lr = learning_rate
        
    def predict(self, X):
        """Make predictions with the model.
        
        Args:
            X: Input matrix of shape (n_samples, input_dim)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias
    
    def loss(self, X, y):
        """Compute mean squared error loss.
        
        Args:
            X: Input matrix of shape (n_samples, input_dim)
            y: Target values of shape (n_samples,)
            
        Returns:
            Mean squared error loss
        """
        y_pred = self.predict(X)
        return np.mean((y_pred - y) ** 2)
    
    def update(self, X, y):
        """Update weights and bias using gradient descent.
        
        Args:
            X: Input matrix of shape (n_samples, input_dim)
            y: Target values of shape (n_samples,)
            
        Returns:
            Current loss after update
        """
        y_pred = self.predict(X)
        error = y_pred - y
        
        # Compute gradients
        dw = (2 / len(y)) * np.dot(X.T, error)
        db = (2 / len(y)) * np.sum(error)
        
        # Update weights and bias
        self.weights -= self.lr * dw
        self.bias -= self.lr * db
        
        return self.loss(X, y)
    
    def save(self, path):
        """Save model parameters to file.
        
        Args:
            path: Path to save the model
        """
        np.savez(path, weights=self.weights, bias=self.bias)
```

## 3. Generate Synthetic Data

Let's generate some synthetic data for our model:

```python
def generate_data(n_samples=100, input_dim=2, noise=0.1):
    """Generate synthetic regression data.
    
    Args:
        n_samples: Number of samples
        input_dim: Input dimension
        noise: Noise level
        
    Returns:
        X: Input matrix of shape (n_samples, input_dim)
        y: Target values of shape (n_samples,)
    """
    # Generate random input data
    X = np.random.randn(n_samples, input_dim)
    
    # Generate target values with noise
    true_weights = np.array([0.5, -0.3]) if input_dim == 2 else np.random.randn(input_dim)
    true_bias = 0.1
    y = np.dot(X, true_weights) + true_bias + noise * np.random.randn(n_samples)
    
    return X, y

# Generate training and validation data
X_train, y_train = generate_data(n_samples=100)
X_val, y_val = generate_data(n_samples=20)
```

## 4. Train the Model with W&B Tracking

Now, let's train our model and track the results with W&B:

```python
# Define our experiment configuration
config = {
    "model": {
        "type": "SimpleLinearModel",
        "input_dim": 2
    },
    "training": {
        "learning_rate": 0.01,
        "epochs": 50,
        "batch_size": 10
    },
    "data": {
        "n_samples": 100,
        "noise": 0.1
    }
}

# Load telemetry configuration from environment
telemetry_cfg = load_telemetry_from_sources()

# Create the model
model = SimpleLinearModel(
    input_dim=config["model"]["input_dim"],
    learning_rate=config["training"]["learning_rate"]
)

# Define paths for artifacts
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(exist_ok=True)
model_path = artifacts_dir / "linear_model.npz"

# Training loop with W&B tracking
with ExperimentLogger(telemetry_cfg, config) as experiment:
    # Track initial metrics
    train_loss = model.loss(X_train, y_train)
    val_loss = model.loss(X_val, y_val)
    
    experiment.log_metrics(
        0,  # Step 0
        train_loss=train_loss,
        val_loss=val_loss
    )
    
    print(f"Initial - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Training loop
    for epoch in range(1, config["training"]["epochs"] + 1):
        # Mini-batch training
        batch_size = config["training"]["batch_size"]
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            # Update model
            batch_loss = model.update(X_batch, y_batch)
        
        # Compute metrics for the epoch
        train_loss = model.loss(X_train, y_train)
        val_loss = model.loss(X_val, y_val)
        
        # Log metrics to W&B
        experiment.log_metrics(
            epoch,  # Current epoch as step
            train_loss=train_loss,
            val_loss=val_loss,
            weights_norm=np.linalg.norm(model.weights),
            bias=float(model.bias)
        )
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save the final model
    model.save(model_path)
    
    # Log the model as an artifact
    experiment.log_artifact(
        path=model_path,
        name="linear_model",
        type="model"
    )
    
    # Set summary metrics
    experiment.set_summary(
        final_train_loss=train_loss,
        final_val_loss=val_loss,
        weights=model.weights.tolist(),
        bias=float(model.bias)
    )
    
print("\nTraining completed. Final model parameters:")
print(f"Weights: {model.weights}")
print(f"Bias: {model.bias}")
```

## 5. Visualizing Results

Let's visualize the model predictions compared to the true data:

```python
# For simplicity, let's visualize the results in 2D (first feature only)
plt.figure(figsize=(10, 6))

# Plot training data
plt.scatter(X_train[:, 0], y_train, alpha=0.6, label='Training Data')

# Plot validation data
plt.scatter(X_val[:, 0], y_val, alpha=0.6, marker='x', label='Validation Data')

# Create a line for model predictions
x_range = np.linspace(min(X_train[:, 0].min(), X_val[:, 0].min()) - 1, 
                     max(X_train[:, 0].max(), X_val[:, 0].max()) + 1, 100)
# Use average value for the second feature for prediction
avg_second_feature = np.mean(X_train[:, 1])
x_for_pred = np.column_stack((x_range, np.ones_like(x_range) * avg_second_feature))
y_pred = model.predict(x_for_pred)

plt.plot(x_range, y_pred, 'r-', label='Model Prediction')

plt.title('Linear Model Predictions')
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.legend()
plt.grid(alpha=0.3)

# Save the figure for W&B logging
plt.savefig('artifacts/prediction_plot.png')
plt.show()
```

## 6. Logging Additional Artifacts

Now let's log the visualization as an artifact:

```python
# Create a new experiment logger
with ExperimentLogger(telemetry_cfg, config) as experiment:
    # Log the visualization
    experiment.log_artifact(
        path='artifacts/prediction_plot.png',
        name='prediction_plot',
        type='plot'
    )
    
    print("Visualization logged as an artifact")
```

## 7. Syncing Offline Runs

If you've been working in offline mode, you can sync your runs to the W&B server when you're ready:

```python
# This cell uses the sync_wandb_runs.py script to sync offline runs
# It will only work if you have a W&B API key set

# !python ../sync_wandb_runs.py --dry-run  # Uncomment to test what would be synced
# !python ../sync_wandb_runs.py  # Uncomment to actually sync
```

## 8. Programmatically Accessing W&B Runs

You can also access your W&B runs programmatically using the W&B API:

```python
# This cell requires wandb to be installed and a W&B API key to be set
# Uncomment to run

# import wandb
# api = wandb.Api()

# # Get a list of runs for your project
# runs = api.runs("ianshank-none/granite-tutorial")

# print(f"Found {len(runs)} runs in the project")
# for run in runs:
#     print(f"Run: {run.name}, ID: {run.id}, Status: {run.state}")
#     print(f"  Summary: {run.summary._json_dict}")
#     print("  -" * 30)
```

## 9. Next Steps

Now that you've learned how to use W&B with the Granite SQE Agent project, here are some next steps to explore:

1. **Run Sweeps**: Use W&B's sweep functionality for hyperparameter optimization
2. **Create Custom Visualizations**: Use W&B's custom visualization features
3. **Compare Different Models**: Run multiple experiments and compare them
4. **Integrate with Real Models**: Apply these techniques to your actual models

For more information, refer to the [WANDB_WORKFLOW.md](../WANDB_WORKFLOW.md) guide.
