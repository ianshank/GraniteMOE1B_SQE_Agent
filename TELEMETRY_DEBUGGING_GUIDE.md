# Telemetry System Debugging Guide

This guide provides detailed instructions for debugging the telemetry system in the Granite SQE Agent. It covers common issues, troubleshooting steps, and how to use the logging capabilities to diagnose problems.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Common Issues](#common-issues)
3. [Debugging Environment Variables](#debugging-environment-variables)
4. [Logging Levels](#logging-levels)
5. [Debugging W&B Integration](#debugging-wandb-integration)
6. [Debugging TensorBoard Integration](#debugging-tensorboard-integration)
7. [Debugging Evaluation Pipeline](#debugging-evaluation-pipeline)
8. [Running the Test Suite](#running-the-test-suite)

## Prerequisites

Ensure you have all required dependencies installed:

```bash
pip install -r requirements.txt
```

For debugging, it's also helpful to install:

```bash
pip install ipdb pytest-xdist pytest-timeout
```

## Common Issues

### W&B Not Logging

If W&B metrics aren't appearing in the dashboard:

1. Check if W&B is properly enabled in your config:
   ```python
   config = TelemetryConfig(enable_wandb=True, wandb_project="your-project")
   ```

2. Verify the environment variables are set:
   ```bash
   export WANDB_PROJECT="your-project"
   export WANDB_ENTITY="your-entity"  # Optional
   ```

3. Check if W&B is in offline mode:
   ```bash
   echo $WANDB_MODE  # Should not be "offline" if you want online logging
   ```

4. Look for W&B directories in the current directory:
   ```bash
   ls -la wandb/
   ```
   
5. Check for error messages in the `wandb/debug.log` file.

### TensorBoard Not Showing Data

If TensorBoard isn't displaying your metrics:

1. Verify TensorBoard is enabled in your config:
   ```python
   config = TelemetryConfig(enable_tensorboard=True, tb_log_dir="runs/")
   ```

2. Check if the log directory was created and contains files:
   ```bash
   ls -la runs/
   ```

3. Make sure TensorBoard is running and pointing to the correct directory:
   ```bash
   tensorboard --logdir=runs
   ```

4. Check for permission issues with the log directory.

### Evaluation Metrics Not Calculating

If evaluation metrics are missing or incorrect:

1. Ensure you're passing the correct `task_type` to the `evaluate()` function.
2. Check that your model output and labels have the same format and length.
3. For classification tasks, verify that class indices match between predictions and targets.
4. For text generation tasks, ensure references are in the correct format (list of lists).

## Debugging Environment Variables

Set these environment variables to enable detailed debugging:

```bash
# Enable debug logging
export PYTHONVERBOSE=1
export GRANITE_LOG_LEVEL=DEBUG

# W&B debugging
export WANDB_DEBUG=true
export WANDB_CONSOLE=on

# TensorBoard debugging
export TF_CPP_MIN_LOG_LEVEL=0
```

## Logging Levels

The telemetry system uses Python's logging module with the following levels:

- `DEBUG`: Detailed information, typically useful only for diagnosing problems
- `INFO`: Confirmation that things are working as expected
- `WARNING`: Indication that something unexpected happened but the process continues
- `ERROR`: Due to a more serious problem, the software hasn't been able to perform a function
- `CRITICAL`: A very serious error, indicating that the program itself may be unable to continue

To set the logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

## Debugging W&B Integration

### Offline Mode

To debug W&B issues without making API calls:

1. Enable offline mode:
   ```bash
   export WANDB_MODE=offline
   ```

2. Run your code normally.

3. Examine the offline run files:
   ```bash
   ls -la wandb/offline-run-*
   ```

4. When ready to sync:
   ```bash
   python sync_wandb_runs.py --run-path wandb/offline-run-*
   ```

### W&B Log Files

Check the W&B logs for errors:

```bash
cat wandb/debug.log
cat wandb/debug-internal.log
```

### Testing W&B Authentication

Test if your API key is valid:

```bash
python -c "import wandb; wandb.login()"
```

### Inspecting W&B Runs

To inspect a specific run:

```python
import wandb
api = wandb.Api()
run = api.run("entity/project/run_id")
print(run.summary)
print(list(run.scan_history()))
```

## Debugging TensorBoard Integration

### Verifying TensorBoard Files

Check if TensorBoard event files are being created:

```bash
find runs/ -name "events.out.tfevents.*"
```

### Running TensorBoard with Verbose Output

```bash
tensorboard --logdir=runs --verbose
```

### Checking TensorBoard File Content

You can use the `tensorboard_logger` utility to read raw event files:

```python
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator("runs/your-run-dir")
ea.Reload()
print(ea.Tags())  # Prints all available tags
print(ea.Scalars('loss'))  # Prints loss values
```

## Debugging Evaluation Pipeline

### Step-by-Step Debugging

To debug evaluation:

1. Enable detailed logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Use manual step execution to trace issues:
   ```python
   predictions = [0, 1, 1]
   targets = [0, 1, 0]
   probabilities = [0.2, 0.8, 0.6]
   
   from src.eval.metrics import compute_classification_metrics
   metrics = compute_classification_metrics(predictions, targets, probabilities)
   print(metrics)
   ```

3. Check for data type or shape issues:
   ```python
   print(f"Predictions type: {type(predictions)}, shape: {getattr(predictions, 'shape', len(predictions))}")
   print(f"Targets type: {type(targets)}, shape: {getattr(targets, 'shape', len(targets))}")
   ```

### Adding Debug Assertions

Add assertions to catch issues early:

```python
assert len(predictions) == len(targets), f"Mismatched lengths: {len(predictions)} vs {len(targets)}"
assert all(isinstance(p, (int, float)) for p in predictions), f"Invalid prediction types: {[type(p) for p in predictions]}"
```

## Running the Test Suite

The test suite includes comprehensive tests for the telemetry system:

```bash
# Run all tests
pytest -v granite-test-generator/tests/

# Run specific test categories
pytest -v granite-test-generator/tests/unit/test_telemetry_config.py
pytest -v granite-test-generator/tests/unit/test_experiment_logger.py
pytest -v granite-test-generator/tests/unit/test_metrics.py
pytest -v granite-test-generator/tests/unit/test_evaluation_helpers.py
pytest -v granite-test-generator/tests/integration/test_end_to_end_workflow.py
pytest -v granite-test-generator/tests/contract/test_wandb_api_contract.py

# Run tests with increased verbosity
pytest -vvs granite-test-generator/tests/unit/test_telemetry_config.py

# Run tests with output logging
pytest -v --log-cli-level=DEBUG granite-test-generator/tests/unit/test_experiment_logger.py
```

### Debugging Test Failures

If tests are failing:

1. Run the failing test with increased verbosity:
   ```bash
   pytest -vvs granite-test-generator/tests/unit/test_metrics.py::TestClassificationMetrics::test_binary_classification
   ```

2. Use the `-xvs` flags to stop at first failure and show verbose output:
   ```bash
   pytest -xvs granite-test-generator/tests/unit/test_metrics.py
   ```

3. Use the `--pdb` flag to drop into a debugger on failure:
   ```bash
   pytest -xvs --pdb granite-test-generator/tests/unit/test_metrics.py
   ```

4. Add a temporary debug print statement in the test to see values:
   ```python
   import pprint
   pprint.pprint(metrics)
   ```

## Advanced Debugging Techniques

For complex issues:

1. **Environment Isolation**: Create a clean environment to test telemetry:
   ```bash
   python -m venv telemetry_debug
   source telemetry_debug/bin/activate
   pip install -r requirements.txt
   ```

2. **Component Isolation**: Test each component separately:
   ```python
   # Test TensorBoard in isolation
   from torch.utils.tensorboard import SummaryWriter
   writer = SummaryWriter("debug_runs")
   writer.add_scalar("test", 1.0, 0)
   writer.close()
   ```

3. **Minimal Reproducible Example**: Create a minimal example to reproduce issues:
   ```python
   from src.config import TelemetryConfig
   from src.telemetry import ExperimentLogger
   
   config = TelemetryConfig(enable_tensorboard=True, tb_log_dir="debug_runs")
   with ExperimentLogger(config, {"test": True}) as logger:
       logger.log_metrics(1, test_metric=0.5)
   ```

By following this debugging guide, you should be able to diagnose and fix most issues with the telemetry system. If you encounter persistent problems, please report them with detailed logs and reproduction steps.
