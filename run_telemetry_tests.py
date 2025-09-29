#!/usr/bin/env python3
"""
Test runner for the telemetry system tests.

This script runs all the telemetry-related tests and generates a report
of the test results, making it easy to validate the telemetry integration.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define test categories and their file patterns
TEST_CATEGORIES = {
    "unit": [
        "tests/unit/test_telemetry_config.py",
        "tests/unit/test_experiment_logger.py",
        "tests/unit/test_metrics.py",
        "tests/unit/test_evaluation_helpers.py",
        "granite-test-generator/tests/unit/test_telemetry_config.py",
        "granite-test-generator/tests/unit/test_experiment_logger.py",
        "granite-test-generator/tests/unit/test_metrics.py", 
        "granite-test-generator/tests/unit/test_evaluation_helpers.py",
    ],
    "integration": [
        "tests/integration/test_end_to_end_workflow.py",
        "granite-test-generator/tests/integration/test_end_to_end_workflow.py",
        "granite-test-generator/tests/integration/test_training_telemetry.py",
    ],
    "contract": [
        "tests/contract/test_wandb_api_contract.py",
        "granite-test-generator/tests/contract/test_wandb_api_contract.py",
    ],
    "all_telemetry": []  # Will be populated with all the above
}

# Populate the "all_telemetry" category
TEST_CATEGORIES["all_telemetry"] = (
    TEST_CATEGORIES["unit"] + 
    TEST_CATEGORIES["integration"] + 
    TEST_CATEGORIES["contract"]
)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run telemetry system tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--category", "-c",
        choices=list(TEST_CATEGORIES.keys()),
        default="all_telemetry",
        help="Test category to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times, e.g., -vv)"
    )
    
    parser.add_argument(
        "--xdist", "-x",
        type=int,
        default=None,
        help="Number of parallel processes for pytest-xdist (requires pytest-xdist)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with PDB on test failures"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level for tests"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_results"),
        help="Directory to store test results"
    )
    
    parser.add_argument(
        "--junit-xml",
        action="store_true",
        help="Generate JUnit XML report"
    )
    
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML report (requires pytest-html)"
    )
    
    parser.add_argument(
        "--env-setup",
        action="store_true",
        help="Set up environment variables for testing"
    )
    
    return parser.parse_args()


def ensure_test_paths_exist(test_paths: List[str]) -> List[str]:
    """Filter the list of test paths to only those that exist."""
    existing_paths = []
    for path in test_paths:
        if os.path.exists(path):
            existing_paths.append(path)
        else:
            logger.warning(f"Test path does not exist: {path}")
    return existing_paths


def setup_test_environment() -> None:
    """Set up environment variables for testing."""
    # Set W&B to offline mode to avoid API calls
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_PROJECT"] = "telemetry-tests"
    
    # Set TensorBoard directory
    os.environ["TB_LOG_DIR"] = "test_runs"
    
    # Set logging level
    os.environ["GRANITE_LOG_LEVEL"] = "DEBUG"
    
    logger.info("Test environment set up with:")
    logger.info(f"WANDB_MODE={os.environ['WANDB_MODE']}")
    logger.info(f"WANDB_PROJECT={os.environ['WANDB_PROJECT']}")
    logger.info(f"TB_LOG_DIR={os.environ['TB_LOG_DIR']}")
    logger.info(f"GRANITE_LOG_LEVEL={os.environ['GRANITE_LOG_LEVEL']}")


def run_tests(
    test_paths: List[str],
    verbosity: int = 0,
    xdist_processes: Optional[int] = None,
    debug: bool = False,
    log_level: str = "INFO",
    output_dir: Optional[Path] = None,
    junit_xml: bool = False,
    html_report: bool = False
) -> Tuple[bool, Dict[str, Union[str, int]]]:
    """
    Run pytest on the specified test paths.
    
    Args:
        test_paths: List of test paths to run
        verbosity: Verbosity level (0-3)
        xdist_processes: Number of parallel processes for pytest-xdist
        debug: Enable PDB debugging on test failures
        log_level: Logging level for tests
        output_dir: Directory to store test results
        junit_xml: Generate JUnit XML report
        html_report: Generate HTML report
    
    Returns:
        Tuple of (success, report_data)
    """
    if not test_paths:
        logger.error("No test paths specified")
        return False, {"error": "No test paths specified"}
    
    # Ensure output directory exists
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add verbosity flags
    verbosity_flags = {
        0: [],
        1: ["-v"],
        2: ["-vv"],
        3: ["-vvs"],
    }
    cmd.extend(verbosity_flags.get(min(verbosity, 3), ["-v"]))
    
    # Add xdist if specified
    if xdist_processes:
        cmd.extend(["-n", str(xdist_processes)])
    
    # Add debug flag if specified
    if debug:
        cmd.extend(["--pdb"])
    
    # Add logging configuration
    cmd.extend(["--log-cli-level", log_level])
    
    # Add reporting options
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if junit_xml and output_dir:
        junit_path = output_dir / f"junit-telemetry-{timestamp}.xml"
        cmd.extend(["--junitxml", str(junit_path)])
    
    if html_report and output_dir:
        html_path = output_dir / f"report-telemetry-{timestamp}.html"
        cmd.extend(["--html", str(html_path), "--self-contained-html"])
    
    # Add test paths
    cmd.extend(test_paths)
    
    # Run the tests
    logger.info(f"Running tests with command: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # We handle the return code ourselves
        )
        success = result.returncode == 0
        
        # Save output if output directory specified
        if output_dir:
            stdout_path = output_dir / f"stdout-{timestamp}.log"
            stderr_path = output_dir / f"stderr-{timestamp}.log"
            
            with open(stdout_path, "w") as f:
                f.write(result.stdout)
            
            with open(stderr_path, "w") as f:
                f.write(result.stderr)
            
            logger.info(f"Test stdout saved to {stdout_path}")
            logger.info(f"Test stderr saved to {stderr_path}")
        
        # Parse test results for report
        end_time = time.time()
        duration = end_time - start_time
        
        # Simple parsing of test summary from output
        report_data = {
            "success": success,
            "duration_seconds": duration,
            "timestamp": timestamp,
            "command": " ".join(cmd),
        }
        
        # Try to extract test summary from output
        summary_lines = [line for line in result.stdout.splitlines() if "passed" in line or "failed" in line or "error" in line or "skipped" in line]
        if summary_lines:
            report_data["summary"] = summary_lines[-1]
        
        return success, report_data
    
    except subprocess.SubprocessError as e:
        logger.error(f"Error running tests: {e}")
        return False, {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return False, {"error": str(e)}


def save_report(report_data: Dict[str, Union[str, int]], output_dir: Path) -> None:
    """Save the test report to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"telemetry_test_report_{report_data.get('timestamp', 'unknown')}.json"
    
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    
    logger.info(f"Test report saved to {report_path}")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Set up logging level
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    
    # Print banner
    logger.info("=" * 80)
    logger.info("TELEMETRY SYSTEM TEST RUNNER".center(80))
    logger.info("=" * 80)
    
    # Set up environment if requested
    if args.env_setup:
        setup_test_environment()
    
    # Get test paths for the selected category
    test_paths = TEST_CATEGORIES[args.category]
    existing_test_paths = ensure_test_paths_exist(test_paths)
    
    if not existing_test_paths:
        logger.error(f"No test paths found for category '{args.category}'")
        return 1
    
    logger.info(f"Running {len(existing_test_paths)} test paths for category '{args.category}'")
    
    # Run the tests
    success, report_data = run_tests(
        existing_test_paths,
        verbosity=args.verbose,
        xdist_processes=args.xdist,
        debug=args.debug,
        log_level=args.log_level,
        output_dir=args.output_dir,
        junit_xml=args.junit_xml,
        html_report=args.html_report
    )
    
    # Save the report
    save_report(report_data, args.output_dir)
    
    # Print summary
    logger.info("=" * 80)
    if success:
        logger.info("✅ ALL TESTS PASSED".center(80))
    else:
        logger.error("❌ TESTS FAILED".center(80))
    logger.info("=" * 80)
    
    if "summary" in report_data:
        logger.info(f"Summary: {report_data['summary']}")
    
    logger.info(f"Duration: {report_data['duration_seconds']:.2f} seconds")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
