#!/usr/bin/env python3
"""
Main entry point for the Granite MoE Test Generator system.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main() -> None:
    """Main function to run the Granite MoE Test Generator.
    
    This is the primary entry point for the granite-test-generator package.
    It initializes the system and runs the complete test case generation workflow.
    
    Raises:
        ImportError: If required modules cannot be imported
        RuntimeError: If system initialization fails
    """
    import logging
    import asyncio
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Granite MoE Test Generator from package entry point")
    logger.debug(f"Python path: {sys.path[0]}")
    logger.debug(f"Working directory: {os.getcwd()}")
    
    try:
        # Import and run the main system
        from src.main import main as granite_main
        asyncio.run(granite_main())
        
        logger.info("Granite MoE Test Generator completed successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Ensure all dependencies are installed and src/ is in Python path")
        raise
    except Exception as e:
        logger.error(f"System execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
