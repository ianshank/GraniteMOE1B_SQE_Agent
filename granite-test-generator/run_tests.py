#!/usr/bin/env python3
"""
Test runner script for the Granite MoE Test Generator project.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all unit tests with pytest."""
    print("Running Granite MoE Test Generator unit tests...")
    print("=" * 60)
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short",
            "--disable-warnings"
        ], cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("\n All tests passed!")
        else:
            print(f"\nTests failed with exit code {result.returncode}")
            
        return result.returncode
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def run_tests_with_coverage():
    """Run tests with coverage report."""
    print("Running tests with coverage report...")
    print("=" * 60)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--cov=src", 
            "--cov-report=term-missing",
            "--tb=short",
            "--disable-warnings"
        ], cwd=Path(__file__).parent)
        
        return result.returncode
        
    except Exception as e:
        print(f"Error running tests with coverage: {e}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--coverage":
        exit_code = run_tests_with_coverage()
    else:
        exit_code = run_tests()
    
    sys.exit(exit_code)
