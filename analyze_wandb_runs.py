#!/usr/bin/env python3
"""
Script to analyze W&B runs and extract metrics.

This script provides utilities to extract and analyze metrics
from W&B runs, which is useful for comparing experiment results.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_wandb_availability() -> bool:
    """Check if the wandb package is available.
    
    Returns:
        bool: True if wandb is available, False otherwise.
    """
    try:
        import wandb
        return True
    except ImportError:
        logger.error("wandb package is not installed.")
        logger.info("Please install it with: pip install wandb")
        return False


def extract_metrics_from_run(
    entity: str, 
    project: str, 
    run_id: str, 
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Extract metrics from a W&B run.
    
    Args:
        entity: W&B entity/username.
        project: W&B project name.
        run_id: W&B run ID.
        metrics: List of metrics to extract (default: all).
        
    Returns:
        Dict[str, Any]: Dictionary containing the extracted metrics.
        
    Raises:
        ImportError: If wandb package is not installed.
        Exception: If extracting metrics fails.
    """
    if not check_wandb_availability():
        raise ImportError("wandb package is required but not installed.")
    
    import wandb
    api = wandb.Api()
    
    try:
        # Get the run
        run_path = f"{entity}/{project}/{run_id}"
        logger.info(f"Fetching run: {run_path}")
        run = api.run(run_path)
        
        # Extract summary metrics
        summary = {k: v for k, v in run.summary.items()}
        
        # Extract config
        config = {k: v for k, v in run.config.items()}
        
        # Extract history for specified metrics
        history = None
        if metrics:
            history_dict = {}
            history = run.scan_history(keys=metrics)
            for row in history:
                for metric in metrics:
                    if metric in row:
                        if metric not in history_dict:
                            history_dict[metric] = []
                        history_dict[metric].append((row.get("_step", 0), row[metric]))
        
        # Prepare result
        result = {
            "run_id": run_id,
            "name": run.name,
            "summary": summary,
            "config": config,
        }
        
        if history is not None:
            result["history"] = history_dict
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract metrics from run {run_id}: {e}")
        raise


def compare_runs(
    entity: str, 
    project: str, 
    run_ids: List[str], 
    metrics: Optional[List[str]] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Compare metrics from multiple W&B runs.
    
    Args:
        entity: W&B entity/username.
        project: W&B project name.
        run_ids: List of W&B run IDs to compare.
        metrics: List of metrics to compare (default: all).
        output_file: Path to save the comparison results (default: None).
        
    Returns:
        Dict[str, Any]: Dictionary containing the comparison results.
        
    Raises:
        ImportError: If wandb package is not installed.
        Exception: If comparing runs fails.
    """
    if not check_wandb_availability():
        raise ImportError("wandb package is required but not installed.")
    
    try:
        # Extract metrics from each run
        runs_data = {}
        for run_id in run_ids:
            runs_data[run_id] = extract_metrics_from_run(entity, project, run_id, metrics)
        
        # Compare summary metrics
        comparison = {
            "summary_comparison": {},
            "config_comparison": {},
        }
        
        # Collect all unique summary metrics and config keys
        all_summary_metrics = set()
        all_config_keys = set()
        
        for run_id, data in runs_data.items():
            all_summary_metrics.update(data["summary"].keys())
            all_config_keys.update(data["config"].keys())
        
        # Compare summary metrics
        for metric in all_summary_metrics:
            comparison["summary_comparison"][metric] = {
                run_id: data["summary"].get(metric, None)
                for run_id, data in runs_data.items()
            }
        
        # Compare config values
        for key in all_config_keys:
            comparison["config_comparison"][key] = {
                run_id: data["config"].get(key, None)
                for run_id, data in runs_data.items()
            }
        
        # Add basic run info
        comparison["runs"] = {
            run_id: {"name": data["name"]}
            for run_id, data in runs_data.items()
        }
        
        # Save comparison results if output file is provided
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(comparison, f, indent=2)
            logger.info(f"Comparison results saved to {output_file}")
        
        return comparison
        
    except Exception as e:
        logger.error(f"Failed to compare runs: {e}")
        raise


def main() -> int:
    """Main entry point for the script.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(description="Analyze W&B runs and extract metrics.")
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity/username (default: from WANDB_ENTITY env var)"
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT"),
        help="W&B project name (default: from WANDB_PROJECT env var)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Extract metrics command
    extract_parser = subparsers.add_parser("extract", help="Extract metrics from a run")
    extract_parser.add_argument("run_id", help="W&B run ID")
    extract_parser.add_argument(
        "--metrics",
        help="Comma-separated list of metrics to extract (default: all)"
    )
    extract_parser.add_argument(
        "--output",
        help="Output file path (default: print to stdout)"
    )
    
    # Compare runs command
    compare_parser = subparsers.add_parser("compare", help="Compare metrics from multiple runs")
    compare_parser.add_argument("run_ids", help="Comma-separated list of W&B run IDs to compare")
    compare_parser.add_argument(
        "--metrics",
        help="Comma-separated list of metrics to compare (default: all)"
    )
    compare_parser.add_argument(
        "--output",
        help="Output file path (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    # Validate entity and project
    if not args.entity:
        logger.error("W&B entity not provided. Use --entity or set WANDB_ENTITY environment variable.")
        return 1
        
    if not args.project:
        logger.error("W&B project not provided. Use --project or set WANDB_PROJECT environment variable.")
        return 1
    
    # Check wandb availability
    if not check_wandb_availability():
        return 1
    
    try:
        if args.command == "extract":
            # Parse metrics if provided
            metrics = args.metrics.split(",") if args.metrics else None
            
            # Extract metrics from run
            result = extract_metrics_from_run(args.entity, args.project, args.run_id, metrics)
            
            # Output results
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Metrics saved to {args.output}")
            else:
                print(json.dumps(result, indent=2))
                
        elif args.command == "compare":
            # Parse run IDs
            run_ids = args.run_ids.split(",")
            
            # Parse metrics if provided
            metrics = args.metrics.split(",") if args.metrics else None
            
            # Compare runs
            result = compare_runs(args.entity, args.project, run_ids, metrics, args.output)
            
            # Output results if not already saved to file
            if not args.output:
                print(json.dumps(result, indent=2))
        
        else:
            logger.error("No command specified. Use one of: extract, compare.")
            parser.print_help()
            return 1
            
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
