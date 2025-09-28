#!/usr/bin/env python3
"""
Main entry point for the Granite MoE Test Generator system.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main function to run the Granite MoE Test Generator."""
    print("Starting Granite MoE Test Generator...")
    print(f"Python path: {sys.path[0]}")
    print(f"Working directory: {os.getcwd()}")
    
    # TODO: Initialize & run the main system components
    # Will be implemented as the system components are developed
    
    print("Granite MoE Test Generator initialized successfully!")

if __name__ == "__main__":
    main()
