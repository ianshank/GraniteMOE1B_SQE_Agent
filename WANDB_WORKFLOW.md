# Weights & Biases Workflow Guide

This guide describes the complete workflow for using Weights & Biases (W&B) with the Granite SQE Agent project for experiment tracking and model evaluation.

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Configuration](#configuration)
4. [Running Experiments](#running-experiments)
5. [Working with Offline Runs](#working-with-offline-runs)
6. [Analyzing Experiments](#analyzing-experiments)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Overview

The Granite SQE Agent project uses Weights & Biases for experiment tracking. Key features include:

- **Experiment Tracking**: Log metrics, parameters, and artifacts
- **Offline Mode**: Work without an internet connection
- **Run Comparison**: Compare different experiments
- **Artifact Management**: Store and version models and datasets

## Setup

### 1. Install Dependencies

```bash
pip install wandb
```

### 2. Set Up W&B API Key

Get your API key from the [W&B website](https://wandb.ai/settings) and set it as an environment variable:

```bash
export WANDB_API_KEY=your-api-key
```

For permanent setup, add it to your `.bashrc` or `.zshrc` file.

### 3. Configure Project Settings

Set default project and entity:

```bash
export WANDB_ENTITY=ianshank-none
export WANDB_PROJECT=QA-Model
```

## Configuration

### Environment Variables

The following environment variables control W&B behavior:

| Variable | Description | Example |
|----------|-------------|---------|
| `WANDB_API_KEY` | Your W&B API key | `export WANDB_API_KEY=abc123` |
| `WANDB_ENTITY` | W&B username or team name | `export WANDB_ENTITY=ianshank-none` |
| `WANDB_PROJECT` | W&B project name | `export WANDB_PROJECT=QA-Model` |
| `WANDB_RUN_NAME` | Name for the current run | `export WANDB_RUN_NAME=experiment-1` |
| `WANDB_TAGS` | Tags for the run (comma-separated) | `export WANDB_TAGS=baseline,test` |
| `WANDB_MODE` | W&B mode (online/offline/disabled) | `export WANDB_MODE=offline` |

### Telemetry Configuration

In the Granite SQE Agent project, W&B integration is handled through the `TelemetryConfig` class in `src/config/telemetry.py`. You can configure W&B via:

1. **Environment Variables**: See above
2. **CLI Arguments**: When running the training script

```bash
python train.py --enable-wandb --wandb-project QA-Model --wandb-tags "baseline,test"
```

3. **.env File**: Copy `.env.sample` to `.env` and modify:

```
ENABLE_WANDB=true
WANDB_PROJECT=QA-Model
WANDB_ENTITY=ianshank-none
```

## Running Experiments

### Training with W&B Enabled

```bash
# Run with W&B tracking enabled
python train.py --enable-wandb --log-checkpoints
```

### Using the ExperimentLogger

For custom scripts, use the `ExperimentLogger` class:

```python
from src.config import load_telemetry_from_sources
from src.telemetry import ExperimentLogger

# Load telemetry config from environment
telemetry_cfg = load_telemetry_from_sources()

# Define your experiment configuration
config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "model_type": "classifier"
}

# Use the ExperimentLogger as a context manager
with ExperimentLogger(telemetry_cfg, config) as experiment:
    # Log metrics during training
    for epoch in range(10):
        loss = train_epoch()
        accuracy = validate()
        
        # Log metrics for this epoch
        experiment.log_metrics(
            epoch, 
            loss=loss, 
            accuracy=accuracy, 
            epoch=epoch
        )
    
    # Log model artifact
    experiment.log_artifact(
        "model.pt", 
        name="final_model",
        type="model"
    )
    
    # Set summary metrics
    experiment.set_summary(
        final_loss=loss,
        final_accuracy=accuracy
    )
```

### Testing W&B Integration

Use the provided test script to verify W&B integration:

```bash
python test_wandb_integration.py
```

## Working with Offline Runs

### Running in Offline Mode

Set the environment variable:

```bash
export WANDB_MODE=offline
```

Or use CLI arguments:

```bash
python train.py --enable-wandb --wandb-mode offline
```

### Syncing Offline Runs

Use the provided sync script:

```bash
# Sync all offline runs
python sync_wandb_runs.py

# Sync a specific run
python sync_wandb_runs.py --run wandb/offline-run-TIMESTAMP-ID

# Dry run to see what would be synced
python sync_wandb_runs.py --dry-run
```

## Analyzing Experiments

### Using the W&B Dashboard

1. Visit [wandb.ai](https://wandb.ai)
2. Navigate to your project
3. View runs, metrics, and artifacts

### Using the Analysis Script

The provided `analyze_wandb_runs.py` script helps extract and compare metrics from W&B runs:

```bash
# Extract metrics from a run
python analyze_wandb_runs.py extract RUN_ID --metrics "loss,accuracy" --output metrics.json

# Compare multiple runs
python analyze_wandb_runs.py compare RUN_ID1,RUN_ID2 --metrics "loss,accuracy" --output comparison.json
```

### Programmatic Access

```python
import wandb
api = wandb.Api()

# Get a specific run
run = api.run("ianshank-none/QA-Model/RUN_ID")

# Get metrics
metrics_df = run.history()

# Get summary metrics
summary = run.summary

# Get config
config = run.config
```

## Best Practices

1. **Consistent Naming**: Use descriptive names for runs
2. **Tag Runs**: Add tags to group related experiments
3. **Log Hyperparameters**: Log all hyperparameters at the start of the run
4. **Save Artifacts**: Save model checkpoints and datasets as W&B artifacts
5. **Group Runs**: Use W&B's sweep feature for hyperparameter searches
6. **Set Description**: Add a description to your run for context
7. **Use Notes**: Add notes to record observations and insights

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   - Ensure `WANDB_API_KEY` is set correctly
   - Try `wandb login` to authenticate

2. **Network Issues**:
   - Switch to offline mode: `export WANDB_MODE=offline`
   - Sync runs later: `python sync_wandb_runs.py`

3. **Missing Metrics**:
   - Verify you're calling `log_metrics()` correctly
   - Check for errors in the W&B debug log

4. **Large Artifacts**:
   - Consider using `wandb.save()` with `policy='now'` for immediate upload
   - Use file references instead of in-memory objects for large data

### Debug Logs

W&B creates debug logs in the `wandb/` directory. Check these for detailed error information:

```bash
cat wandb/debug.log
```

### Support

For additional help:
- [W&B Documentation](https://docs.wandb.ai/)
- [W&B Community Forum](https://community.wandb.ai/)

---

This guide covers the complete workflow for using Weights & Biases with the Granite SQE Agent project. For project-specific questions, refer to the project documentation or contact the project maintainers.
