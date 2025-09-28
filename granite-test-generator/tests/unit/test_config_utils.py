import pytest
import os
from unittest.mock import patch
from src.utils import config_utils

@pytest.fixture
def sample_config():
    """Provides a sample configuration dictionary for testing."""
    return {
        "api_key": "${API_KEY}",
        "api_secret": "${API_SECRET:default_secret}",
        "timeout": 30,
        "nested": {
            "user": "${DB_USER}",
            "password": "${DB_PASSWORD:admin}"
        },
        "list_of_values": [
            "value1",
            "${LIST_VAR}"
        ]
    }

def test_basic_substitution():
    """Test successful substitution of a required environment variable."""
    config = {"api_key": "${API_KEY}"}
    with patch.dict(os.environ, {"API_KEY": "my_secret_key"}):
        resolved_config = config_utils.resolve_env_vars(config)
        assert resolved_config["api_key"] == "my_secret_key"

def test_default_value_substitution():
    """Test substitution with a default value when the env var is not set."""
    config = {"api_secret": "${API_SECRET:default_secret}"}
    with patch.dict(os.environ, {}, clear=True):
        resolved_config = config_utils.resolve_env_vars(config)
        assert resolved_config["api_secret"] == "default_secret"

def test_default_value_is_overridden():
    """Test that an environment variable overrides a configured default value."""
    config = {"api_secret": "${API_SECRET:default_secret}"}
    with patch.dict(os.environ, {"API_SECRET": "overridden_secret"}):
        resolved_config = config_utils.resolve_env_vars(config)
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
        resolved_config = config_utils.resolve_env_vars(config)
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
        resolved_config = config_utils.resolve_env_vars(config)
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
        resolved_config = config_utils.resolve_env_vars(config)
        assert resolved_config["list_of_values"][1] == "substituted_value"

def test_missing_required_variable_raises_error():
    """Test that a missing required environment variable raises a ValueError."""
    config = {"required_var": "${REQUIRED_VAR}"}
    # Ensure environment is empty
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as excinfo:
            config_utils.resolve_env_vars(config)
        assert "Missing required environment variable: REQUIRED_VAR" in str(excinfo.value)

def test_no_substitution_if_no_variable():
    """Test that values without the ${...} syntax are untouched."""
    config = {"timeout": 30, "string_value": "plain text"}
    with patch.dict(os.environ, {}, clear=True):
        resolved_config = config_utils.resolve_env_vars(config)
        assert resolved_config["timeout"] == 30
        assert resolved_config["string_value"] == "plain text"

def test_empty_config_returns_empty():
    """Test that an empty configuration dictionary is handled gracefully."""
    assert config_utils.resolve_env_vars({}) == {}

def test_non_string_values_are_untouched():
    """Test that non-string values are not processed."""
    config = {"number": 123, "boolean": True, "none": None}
    resolved_config = config_utils.resolve_env_vars(config)
    assert resolved_config == config

def test_malformed_variable_is_ignored():
    """Test that malformed variable syntax is ignored."""
    config = {"key": "${MALFORMED"}
    resolved_config = config_utils.resolve_env_vars(config)
    assert resolved_config["key"] == "${MALFORMED"
