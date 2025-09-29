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
    
    Initializes and runs the complete test case generation workflow using
    the GraniteTestCaseGenerator from the granite-test-generator package.
    
    Raises:
        ImportError: If granite-test-generator components cannot be imported
        RuntimeError: If system initialization fails
    """
    import logging
    import asyncio
    
    # Configure logging for main entry point
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Granite MoE Test Generator")
    logger.debug(f"Python path: {sys.path[0]}")
    logger.debug(f"Working directory: {os.getcwd()}")
    
    try:
        # Resolve integration config path to an absolute path before changing directory
        integ_path = os.getenv("INTEGRATION_CONFIG_PATH")
        if integ_path and not Path(integ_path).is_absolute():
            abs_integ_path = str(Path(integ_path).resolve())
            os.environ["INTEGRATION_CONFIG_PATH"] = abs_integ_path
            logger.info(f"Resolved relative INTEGRATION_CONFIG_PATH to absolute path: {abs_integ_path}")
            
        # Determine granite-test-generator directory for proper path resolution
        project_root_override = os.getenv("GRANITE_PROJECT_ROOT")
        if project_root_override:
            granite_dir = Path(project_root_override).expanduser()
        else:
            granite_dir = Path(__file__).parent / "granite-test-generator"

        granite_dir = granite_dir.resolve()
        logger.debug("Resolved granite project directory to %s", granite_dir)
        if not granite_dir.exists():
            raise RuntimeError(f"Granite test generator directory not found: {granite_dir}")

        os.chdir(granite_dir)
        logger.info("Changed working directory to: %s", granite_dir)
        
        # Import and run the main system
        from src.main import main as granite_main
        
        # Parse command line arguments
        import argparse
        import sys
        parser = argparse.ArgumentParser(description="Granite Test Case Generator")
        parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Path to configuration file")
        parser.add_argument("--multiple-suites", action="store_true", help="Generate multiple test suites (functional, regression, E2E)")
        
        # In test environments, we might have pytest arguments like '-q tests'
        # Only parse known args to avoid errors in test environments
        args, unknown = parser.parse_known_args()
        
        # If running under pytest, log the unknown arguments
        if unknown:
            logger.debug(f"Ignoring unknown arguments for testing: {unknown}")
        
        # Run the main function with command line arguments
        asyncio.run(granite_main(
            config_path=args.config,
            generate_multiple_suites=args.multiple_suites
        ))
        
        logger.info("Granite MoE Test Generator completed successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import granite-test-generator components: {e}")
        logger.error("Ensure you're running from the project root directory")
        raise
    except Exception as e:
        logger.error(f"System execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
