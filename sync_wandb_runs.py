#!/usr/bin/env python3
"""
Script to sync offline W&B runs to the W&B server.

This script provides a convenient way to sync offline W&B runs
to the W&B server when you're ready to share your results.
"""

import argparse
import glob
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_wandb_api_key() -> bool:
    """Check if the W&B API key is set in the environment.
    
    Returns:
        bool: True if the API key is set, False otherwise.
    """
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        logger.error("WANDB_API_KEY environment variable is not set.")
        logger.info(
            "Please set your W&B API key using: "
            "export WANDB_API_KEY=your-api-key"
        )
        return False
    return True


def find_offline_runs(pattern: str = "wandb/offline-run-*") -> List[str]:
    """Find offline W&B runs matching the given pattern.
    
    Args:
        pattern: Glob pattern for offline run directories.
        
    Returns:
        List[str]: List of paths to offline run directories.
    """
    offline_dirs = glob.glob(pattern)
    if not offline_dirs:
        logger.warning(f"No offline runs found matching pattern: {pattern}")
    else:
        logger.info(f"Found {len(offline_dirs)} offline runs.")
    return offline_dirs


def sync_run(run_dir: str, dry_run: bool = False) -> bool:
    """Sync a single offline run to the W&B server.
    
    Args:
        run_dir: Path to offline run directory.
        
    Returns:
        bool: True if sync was successful, False otherwise.
    """
    logger.info(f"Syncing run: {run_dir}")
    try:
        if dry_run:
            logger.info(f"[DRY RUN] Would sync run: {run_dir}")
            return True
        else:
            process = subprocess.run(
                ["wandb", "sync", run_dir],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Successfully synced run: {run_dir}")
            logger.debug(process.stdout)
            return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to sync run {run_dir}: {e.stderr}")
        return False


def sync_all_runs(pattern: str = "wandb/offline-run-*", dry_run: bool = False) -> int:
    """Sync all offline runs matching the given pattern.
    
    Args:
        pattern: Glob pattern for offline run directories.
        dry_run: If True, don't actually sync runs, just show what would be done.
        
    Returns:
        int: Number of successfully synced runs.
    """
    # Skip API key check in dry run mode
    if not dry_run and not check_wandb_api_key():
        return 0
        
    offline_dirs = find_offline_runs(pattern)
    if not offline_dirs:
        return 0
    
    successful_syncs = 0
    for run_dir in offline_dirs:
        if sync_run(run_dir, dry_run):
            successful_syncs += 1
    
    logger.info(f"Successfully synced {successful_syncs} out of {len(offline_dirs)} runs.")
    return successful_syncs


def main() -> int:
    """Main entry point for the script.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(description="Sync offline W&B runs to the W&B server.")
    parser.add_argument(
        "--pattern",
        default="wandb/offline-run-*",
        help="Glob pattern for offline run directories (default: wandb/offline-run-*)"
    )
    parser.add_argument(
        "--run",
        help="Sync a specific run directory (overrides --pattern)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually syncing runs"
    )
    
    args = parser.parse_args()
    
    dry_run = args.dry_run
    
    if args.run:
        if not os.path.isdir(args.run):
            logger.error(f"Run directory not found: {args.run}")
            return 1
        # Skip API key check in dry run mode
        if not dry_run and not check_wandb_api_key():
            return 1
        return 0 if sync_run(args.run, dry_run) else 1
    else:
        # Skip API key check in dry run mode
        if not dry_run and not check_wandb_api_key():
            return 1
        synced = sync_all_runs(args.pattern, dry_run)
        return 0 if synced > 0 or dry_run else 1


if __name__ == "__main__":
    sys.exit(main())
