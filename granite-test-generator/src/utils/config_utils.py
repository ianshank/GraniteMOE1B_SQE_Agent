"""
Configuration utilities for handling environment variables and configuration management.

This module provides functions for resolving environment variables in configuration
dictionaries, supporting both required variables (${VAR}) and variables with default
values (${VAR:default}).
"""

import os
import re
import logging
from typing import Dict, List, Any, Union, Optional

logger = logging.getLogger(__name__)

# Regular expression to match environment variable patterns
# Captures: ${VAR} or ${VAR:default}
ENV_VAR_PATTERN = re.compile(r'\${([^:}]+)(?::([^}]+))?}')


def resolve_env_vars(config: Union[Dict[str, Any], List[Any], str, Any]) -> Any:
    """
    Recursively resolves environment variables in a configuration structure.
    
    Supports both required variables (${VAR}) and variables with default values (${VAR:default}).
    If a required variable is not found in the environment, a ValueError is raised.
    
    Args:
        config: Configuration structure (dict, list, or primitive value)
        
    Returns:
        Configuration with all environment variables resolved
        
    Raises:
        ValueError: If a required environment variable is not set
    """
    if isinstance(config, dict):
        return {k: resolve_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [resolve_env_vars(item) for item in config]
    elif isinstance(config, str):
        return _resolve_env_var_in_string(config)
    else:
        # Return non-string primitives unchanged
        return config


def _resolve_env_var_in_string(value: str) -> str:
    """
    Resolves environment variables in a string value.
    
    Args:
        value: String that may contain environment variable references
        
    Returns:
        String with environment variables resolved
        
    Raises:
        ValueError: If a required environment variable is not set
    """
    def _replace_env_var(match):
        var_name = match.group(1)
        default_value = match.group(2)  # Will be None if no default provided
        
        env_value = os.environ.get(var_name)
        
        if env_value is not None:
            logger.debug("Resolved environment variable %s", var_name)
            return env_value
        elif default_value is not None:
            logger.debug("Using default value for environment variable %s", var_name)
            return default_value
        else:
            # No environment variable and no default - this is an error
            error_msg = f"Missing required environment variable: {var_name}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Replace all environment variable patterns in the string
    try:
        return ENV_VAR_PATTERN.sub(_replace_env_var, value)
    except ValueError:
        # Re-raise ValueError from _replace_env_var
        raise
    except Exception as e:
        # Log and re-raise any other exceptions
        logger.error("Error resolving environment variables in string: %s", str(e))
        raise


def load_config_with_env_vars(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file and resolves environment variables.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary with environment variables resolved
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValueError: If a required environment variable is not set
    """
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML is required for loading YAML configuration files")
        raise ImportError("PyYAML is required for loading YAML configuration files")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        logger.debug("Loaded configuration from %s", config_path)
        
        # Resolve environment variables in the loaded configuration
        resolved_config = resolve_env_vars(config)
        return resolved_config
    except FileNotFoundError:
        logger.warning("Configuration file not found: %s", config_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML configuration: %s", str(e))
        raise
    except ValueError:
        # Re-raise ValueError from resolve_env_vars
        raise
    except Exception as e:
        logger.error("Unexpected error loading configuration: %s", str(e))
        raise
