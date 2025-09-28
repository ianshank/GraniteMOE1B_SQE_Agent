"""
Configuration utilities for handling environment variables and configuration loading.
"""

import os
import re
import yaml
from typing import Any, Dict, Union
from pathlib import Path


def substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute environment variables in configuration values.
    
    Supports formats:
    - ${VAR_NAME} - Required variable (raises error if not found)
    - ${VAR_NAME:default_value} - Optional with default
    
    Args:
        value: Configuration value (string, dict, list, or other)
        
    Returns:
        Value with environment variables substituted
        
    Raises:
        ValueError: If required environment variable is not found
    """
    if isinstance(value, str):
        # Find all ${...} patterns
        pattern = r'\$\{([^:}]+)(?::([^}]*))?\}'
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2)
            
            env_value = os.environ.get(var_name)
            
            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                raise ValueError(f"Required environment variable '{var_name}' not found")
        
        return re.sub(pattern, replace_var, value)
    
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    
    else:
        return value


def load_config_with_env(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file with environment variable substitution.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary with environment variables substituted
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If required environment variables are missing
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        return {}
    
    # Substitute environment variables
    return substitute_env_vars(raw_config)


def validate_required_env_vars(required_vars: list) -> Dict[str, str]:
    """
    Validate that required environment variables are set.
    
    Args:
        required_vars: List of required environment variable names
        
    Returns:
        Dictionary of environment variable names to values
        
    Raises:
        ValueError: If any required variables are missing
    """
    missing_vars = []
    env_values = {}
    
    for var_name in required_vars:
        value = os.environ.get(var_name)
        if value is None:
            missing_vars.append(var_name)
        else:
            env_values[var_name] = value
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"Please set these variables or create a .env file."
        )
    
    return env_values


def load_env_file(env_path: Union[str, Path] = '.env') -> Dict[str, str]:
    """
    Load environment variables from a .env file.
    
    Args:
        env_path: Path to .env file (default: '.env')
        
    Returns:
        Dictionary of environment variables loaded
    """
    env_path = Path(env_path)
    env_vars = {}
    
    if not env_path.exists():
        return env_vars
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE format
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                env_vars[key] = value
                # Also set in os.environ for immediate use
                os.environ[key] = value
            else:
                print(f"Warning: Invalid .env format at line {line_num}: {line}")
    
    return env_vars


def get_config_with_defaults(config_path: Union[str, Path], 
                           env_file: Union[str, Path] = '.env') -> Dict[str, Any]:
    """
    Load configuration with environment variable substitution and defaults.
    
    This is the main function to use for loading configuration in the application.
    
    Args:
        config_path: Path to YAML configuration file
        env_file: Path to .env file (default: '.env')
        
    Returns:
        Configuration dictionary with all substitutions applied
    """
    # Load .env file first
    load_env_file(env_file)
    
    # Load and process configuration
    try:
        config = load_config_with_env(config_path)
        return config
    except Exception as e:
        print(f"Warning: Failed to load configuration from {config_path}: {e}")
        return {}