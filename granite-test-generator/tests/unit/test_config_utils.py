import pytest
import os
import yaml
from unittest.mock import patch, mock_open
from src.utils.config_utils import substitute_env_vars, load_config_with_env, load_env_file, validate_required_env_vars, get_config_with_defaults
from src.main import GraniteTestCaseGenerator

def test_basic_substitution():
    """Test successful substitution of a required environment variable."""
    config = {"api_key": "${API_KEY}"}
    with patch.dict(os.environ, {"API_KEY": "my_secret_key"}):
        resolved_config = substitute_env_vars(config)
        assert resolved_config["api_key"] == "my_secret_key"

def test_default_value_substitution():
    """Test substitution with a default value when the env var is not set."""
    config = {"api_secret": "${API_SECRET:default_secret}"}
    with patch.dict(os.environ, {}, clear=True):
        resolved_config = substitute_env_vars(config)
        assert resolved_config["api_secret"] == "default_secret"

def test_default_value_is_overridden():
    """Test that an environment variable overrides a configured default value."""
    config = {"api_secret": "${API_SECRET:default_secret}"}
    with patch.dict(os.environ, {"API_SECRET": "overridden_secret"}):
        resolved_config = substitute_env_vars(config)
        assert resolved_config["api_secret"] == "overridden_secret"

def test_nested_substitution():
    """Test substitution within nested dictionaries."""
    config = {
        "nested": {
            "user": "${DB_USER}",
            "password": "${DB_PASSWORD}"
        }
    }
    env_vars = {"DB_USER": "testuser", "DB_PASSWORD": "testpassword"}
    with patch.dict(os.environ, env_vars):
        resolved_config = substitute_env_vars(config)
        assert resolved_config["nested"]["user"] == "testuser"
        assert resolved_config["nested"]["password"] == "testpassword"

def test_nested_substitution_with_default():
    """Test default value substitution within nested dictionaries."""
    config = {
        "nested": {
            "user": "${DB_USER}",
            "password": "${DB_PASSWORD:admin}"
        }
    }
    with patch.dict(os.environ, {"DB_USER": "testuser"}, clear=True):
        resolved_config = substitute_env_vars(config)
        assert resolved_config["nested"]["user"] == "testuser"
        assert resolved_config["nested"]["password"] == "admin"

def test_list_substitution():
    """Test substitution within a list."""
    config = {
        "list_of_values": [
            "value1",
            "${LIST_VAR}"
        ]
    }
    with patch.dict(os.environ, {"LIST_VAR": "substituted_value"}):
        resolved_config = substitute_env_vars(config)
        assert resolved_config["list_of_values"][1] == "substituted_value"

def test_missing_required_variable_raises_error():
    """Test that a missing required environment variable raises a ValueError."""
    config = {"required_var": "${REQUIRED_VAR}"}
    # Ensure environment is empty
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as excinfo:
            substitute_env_vars(config)
        assert "Required environment variable 'REQUIRED_VAR' not found" in str(excinfo.value)

def test_no_substitution_if_no_variable():
    """Test that values without the ${...} syntax are untouched."""
    config = {"timeout": 30, "string_value": "plain text"}
    with patch.dict(os.environ, {}, clear=True):
        resolved_config = substitute_env_vars(config)
        assert resolved_config["timeout"] == 30
        assert resolved_config["string_value"] == "plain text"

def test_empty_config_returns_empty():
    """Test that an empty configuration dictionary is handled gracefully."""
    assert substitute_env_vars({}) == {}

def test_non_string_values_are_untouched():
    """Test that non-string values are not processed."""
    config = {"number": 123, "boolean": True, "none": None}
    resolved_config = substitute_env_vars(config)
    assert resolved_config == config

def test_malformed_variable_is_ignored():
    """Test that malformed variable syntax is ignored."""
    config = {"key": "${MALFORMED"}
    resolved_config = substitute_env_vars(config)
    assert resolved_config["key"] == "${MALFORMED"

def test_load_config_with_env_vars():
    """Test loading a configuration file with environment variables."""
    yaml_content = """
    api_key: ${API_KEY}
    api_secret: ${API_SECRET:default_secret}
    """
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch.dict(os.environ, {"API_KEY": "my_secret_key"}):
                config = load_config_with_env("config.yaml")
                assert config["api_key"] == "my_secret_key"
                assert config["api_secret"] == "default_secret"

def test_load_config_with_file_not_found():
    """Test that load_config_with_env raises FileNotFoundError when file is not found."""
    with pytest.raises(FileNotFoundError):
        load_config_with_env("nonexistent.yaml")

def test_load_config_with_invalid_yaml():
    """Test that load_config_with_env raises YAMLError when YAML is invalid."""
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="invalid: yaml: content:")):
            with patch("yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML")):
                with pytest.raises(yaml.YAMLError):
                    load_config_with_env("invalid.yaml")

# Tests for the _load_config method in GraniteTestCaseGenerator

def test_granite_load_config_with_dict():
    """Test that _load_config uses the provided dictionary."""
    config_dict = {"test": "value"}
    with patch("src.utils.config_utils.substitute_env_vars", return_value={"test": "resolved"}):
        generator = GraniteTestCaseGenerator(config_dict=config_dict)
        assert generator.config == {"test": "resolved"}

def test_granite_load_config_with_file_not_found():
    """Test that _load_config returns empty dict when file is not found."""
    with patch("src.utils.config_utils.load_config_with_env", side_effect=FileNotFoundError()):
        generator = GraniteTestCaseGenerator(config_path="nonexistent.yaml")
        assert generator.config == {}

def test_granite_load_config_with_invalid_yaml():
    """Test that _load_config returns empty dict when YAML is invalid."""
    with patch("src.utils.config_utils.load_config_with_env", side_effect=yaml.YAMLError("Invalid YAML")):
        generator = GraniteTestCaseGenerator(config_path="invalid.yaml")
        assert generator.config == {}

def test_granite_load_config_with_missing_env_var():
    """Test that _load_config returns empty dict when environment variable is missing."""
    with patch("src.utils.config_utils.load_config_with_env", side_effect=ValueError("Missing required environment variable")):
        generator = GraniteTestCaseGenerator(config_path="config.yaml")
        assert generator.config == {}

def test_granite_load_config_with_generic_exception():
    """Test that _load_config returns empty dict on any other exception."""
    with patch("src.utils.config_utils.load_config_with_env", side_effect=Exception("Unexpected error")):
        generator = GraniteTestCaseGenerator(config_path="config.yaml")
        assert generator.config == {}
