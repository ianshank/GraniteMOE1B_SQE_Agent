# Granite SQE Agent

This repository contains the Granite Software Quality Engineering agent, training harness, and supporting utilities for generating and evaluating test artefacts. The project now includes opt-in telemetry instrumentation so experiments can be tracked with Weights & Biases (W&B) and visualised in TensorBoard.

## Telemetry & Evaluation

Telemetry is disabled by default. Enable it per run via CLI flags or environment variables. The configuration is managed through `TelemetryConfig` (see `src/config/telemetry.py`) and used by the training harness (`train.py`) and evaluation pipeline (`src/eval/evaluate.py`).

### CLI Flags

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
- `--log-checkpoints` to upload model checkpoints when telemetry is enabled

### Environment Variables

Environment variables offer the same controls and remain opt-in:

- `ENABLE_WANDB`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME`, `WANDB_TAGS`
- `ENABLE_TENSORBOARD`, `TB_LOG_DIR`
- `LOG_INTERVAL_STEPS` to adjust telemetry cadence

Copy `.env.sample` to `.env` for local development, but never commit secrets. Configure `WANDB_API_KEY` through your shell or CI secrets store.

### TensorBoard

Launch TensorBoard with the provided script or Makefile target:

```bash
make tb
# or
./scripts/tensorboard.sh
```

### Evaluation Pipeline

The reusable evaluation utilities live under `src/eval/`. Use `evaluate` to compute metrics for classification, regression, or text-generation tasks and automatically persist an `eval_report.json`. When telemetry is enabled, the report is uploaded as a run artifact.

## GitHub Actions Notes

When enabling telemetry in CI:

1. Add a repository secret named `WANDB_API_KEY`.
2. Export it in the workflow step before running training:

   ```yaml
   - name: Configure W&B
     run: |
       echo "WANDB_API_KEY=${WANDB_API_KEY}" >> "$GITHUB_ENV"
     env:
       WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
   ```
3. Opt into telemetry with CLI flags or environment variables in the training step.

Upload artifacts (e.g., `artifacts/eval/eval_report.json` or checkpoints) with `actions/upload-artifact` if you need them outside of W&B.

## Example Runs

```bash
# Run the synthetic training loop without telemetry
python train.py --task-type classification --epochs 1

# Enable W&B with tags and checkpoint uploads
WANDB_PROJECT=granite-experiments ENABLE_WANDB=true \
python train.py --task-type regression --epochs 2 --log-checkpoints

# Evaluate an existing model via the helper (imports evaluate())
python -c "from src.eval.evaluate import evaluate; print('See tests for examples')"
```

Refer to the docstrings in `src/telemetry/experiment.py` and `src/eval/metrics.py` for extension guidelines.
