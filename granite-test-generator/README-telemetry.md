# Telemetry & Evaluation Framework

This section describes the telemetry and evaluation capabilities added to the Granite Test Generator. These features enable experiment tracking with Weights & Biases (W&B) and TensorBoard, as well as comprehensive model evaluation.

## Telemetry Configuration

Telemetry is **disabled by default** and must be explicitly enabled. You can configure it through environment variables or CLI arguments when running training.

### Environment Variables

Copy `.env.sample` to `.env` to configure telemetry options:

```bash
# Enable Weights & Biases tracking
ENABLE_WANDB=true
WANDB_PROJECT=granite-moe-experiments
WANDB_ENTITY=quality-engineering
WANDB_RUN_NAME=granite-trial
WANDB_TAGS=baseline,granite

# Enable TensorBoard logging
ENABLE_TENSORBOARD=true
TB_LOG_DIR=./runs

# Override telemetry logging interval (steps)
LOG_INTERVAL_STEPS=25
```

For W&B authentication, set up your API key:

```bash
export WANDB_API_KEY="your-api-key-here"
```

### CLI Flags

When using the training harness, you can enable telemetry via command-line flags:

```bash
# Enable both W&B and TensorBoard
python train.py --task-type classification --epochs 2 \
  --enable-wandb --wandb-project granite-moe \
  --enable-tensorboard --tb-log-dir ./runs/demo
```

Key flags:

- `--enable-wandb` / `--disable-wandb`
- `--wandb-project`, `--wandb-entity`, `--wandb-run-name`, `--wandb-tags`
- `--enable-tensorboard` / `--disable-tensorboard`
- `--tb-log-dir`
- `--log-checkpoints` to upload model checkpoints

## TensorBoard Usage

Launch TensorBoard to visualize metrics:

```bash
# Using the script
./scripts/tensorboard.sh

# Or directly
tensorboard --logdir runs/ --port 6006 --host 0.0.0.0
```

## Evaluation Framework

The evaluation framework provides reusable metrics computation for:

- **Classification**: Accuracy, precision, recall, F1 (macro/micro), AUROC
- **Regression**: MAE, RMSE, R²
- **Text Generation**: BLEU, ROUGE, exact match, latency

### Using the Evaluation Module

```python
from granite_test_generator.src.eval import evaluate

# Run evaluation and save metrics
metrics = evaluate(
    model=my_model,
    dataloader=eval_loader,
    task_type="classification",  # or "regression"/"text"
    experiment_logger=logger,    # optional, for telemetry
    output_dir="artifacts/eval"
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## Training with Telemetry

The updated training harness supports telemetry integration:

```python
from granite_test_generator.src.config import load_telemetry_from_sources
from granite_test_generator.src.telemetry import ExperimentLogger

# Load telemetry config from environment variables or CLI arguments
telemetry_cfg = load_telemetry_from_sources(args)

with ExperimentLogger(telemetry_cfg, config_snapshot) as experiment:
    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Log metrics
        experiment.log_metrics(step, loss=0.1, accuracy=0.95)
        
        # Log hyperparameters
        experiment.log_params(learning_rate=0.001, batch_size=32)
        
        # Evaluate and log metrics
        metrics = evaluate(model, eval_loader, "classification", 
                          experiment_logger=experiment)
        
    # Log artifacts (like model checkpoints)
    if log_checkpoints:
        experiment.log_artifact("path/to/model.pt", name="checkpoint", type="model")
```

## GitHub Actions Integration

When integrating with GitHub Actions:

1. Add a repository secret named `WANDB_API_KEY`
2. Set up the workflow environment:

```yaml
- name: Configure W&B
  run: |
    echo "WANDB_API_KEY=${WANDB_API_KEY}" >> "$GITHUB_ENV"
  env:
    WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
```

## Troubleshooting

- If W&B connection fails, try using `WANDB_MODE=offline` for local logging
- TensorBoard logs are saved even if the server isn't running
- Missing dependencies will cause graceful degradation - check logs for warnings
