# Syncing Weights & Biases Offline Runs

This guide explains how to sync your offline W&B runs to the cloud.

## Step 1: Authenticate with W&B

First, authenticate with your W&B account:

```bash
wandb login
```

You'll be prompted to enter your API key. Alternatively, you can set the API key directly:

```bash
export WANDB_API_KEY=your-api-key
```

## Step 2: Sync Offline Runs

To sync a specific offline run to the cloud:

```bash
wandb sync wandb/offline-run-TIMESTAMP-ID
```

For example, from your latest run:

```bash
wandb sync wandb/offline-run-20250928_203558-rqfpider
```

## Step 3: Sync All Offline Runs

To sync all offline runs:

```bash
wandb sync wandb/offline-run-*
```

## Step 4: Switch to Online Mode

For future runs, you can switch to online mode:

1. Set the environment variable:
   ```bash
   export WANDB_MODE=online
   ```

2. Or modify your .env file:
   ```
   WANDB_MODE=online
   ```

3. Or use the CLI flag:
   ```bash
   python train.py --enable-wandb --wandb-project QA-Model
   ```

## Step 5: View Your Runs

Once synced, view your runs at:
https://wandb.ai/ianshank-none/QA-Model

## Security Note

Remember to protect your API key and never commit it to version control.
