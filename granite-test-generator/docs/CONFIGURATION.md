# Configuration Management System

This document describes the configuration management system for the Granite Test Generator application.

## Overview

The configuration system provides a flexible way to configure the application through:

1. YAML configuration files
2. Environment variables
3. Command-line arguments

Configuration values can be overridden in the following order of precedence (highest to lowest):
1. Command-line arguments
2. Environment variables
3. Integration configuration file
4. Base configuration file

## Environment Variable Substitution

The system supports substituting environment variables in configuration files using the `${VAR}` syntax.
This allows sensitive information like API tokens to be stored securely in environment variables
rather than in configuration files.

### Syntax

- `${VAR}`: Required environment variable. If not set, an error will be raised.
- `${VAR:default}`: Optional environment variable with default value. If not set, the default value will be used.

### Examples

```yaml
github:
  repo_owner: "my-org"
  repo_name: "my-repo"
  token: "${GITHUB_TOKEN}"  # Required environment variable

jira:
  base_url: "${JIRA_URL:https://my-company.atlassian.net}"  # Optional with default
  username: "${JIRA_USERNAME}"
  api_token: "${JIRA_API_TOKEN}"
```

## Constants

Common configuration values that were previously hardcoded throughout the codebase have been
centralized in `src/utils/constants.py`. This makes the code more maintainable and easier to configure.

### Available Constants

- **HTTP and API related**:
  - `DEFAULT_HTTP_TIMEOUT`: Default timeout for HTTP requests (30 seconds)
  - `DEFAULT_MAX_RETRIES`: Default number of retries for HTTP requests (3)
  - `DEFAULT_BACKOFF_FACTOR`: Default backoff factor for retries (0.5)
  - `DEFAULT_BATCH_SIZE`: Default batch size for processing (4)

- **API Base URLs**:
  - `GITHUB_API_BASE_URL`: Base URL for GitHub API
  - `GITHUB_DEFAULT_HEADERS`: Default headers for GitHub API requests
  - `JIRA_DEFAULT_HEADERS`: Default headers for Jira API requests

- **File and path related**:
  - `DEFAULT_OUTPUT_DIR`: Default output directory for generated test cases
  - `DEFAULT_REQUIREMENTS_DIR`: Default directory for requirements
  - `DEFAULT_USER_STORIES_DIR`: Default directory for user stories
  - `DEFAULT_CACHE_DIR`: Default directory for cache files

- **Model related**:
  - `DEFAULT_MODEL_NAME`: Default model name
  - `DEFAULT_EMBEDDING_MODEL`: Default embedding model
  - `DEFAULT_CHUNK_SIZE`: Default chunk size for text processing
  - `DEFAULT_CHUNK_OVERLAP`: Default chunk overlap for text processing
  - `DEFAULT_TOP_K`: Default number of top results to return
  - `DEFAULT_MAX_TOKENS`: Default maximum tokens for model generation

- **Environment variable names**:
  - `ENV_INTEGRATION_CONFIG_PATH`: Environment variable for integration config path
  - `ENV_LOCAL_ONLY_MODE`: Environment variable for local-only mode
  - `ENV_CONFIG_OVERRIDE_MODE`: Environment variable for config override mode

## Configuration Files

### Base Configuration

The base configuration file (`config/model_config.yaml`) contains the default configuration for the application.

### Integration Configuration

The integration configuration file (specified by the `INTEGRATION_CONFIG_PATH` environment variable)
can override or extend the base configuration. This is useful for environment-specific configurations.

#### Override Modes

The `GRANITE_CONFIG_OVERRIDE_MODE` environment variable controls how the integration configuration
is applied to the base configuration:

- `merge` (default): Merge the integration configuration with the base configuration. Teams from both configurations will be included, with integration teams taking precedence for duplicate names.
- `replace`: Replace the base configuration with the integration configuration. Only teams from the integration configuration will be used.

## Local-Only Mode

Setting the `GRANITE_LOCAL_ONLY` environment variable to `true`, `1`, `yes`, or `on` will force
all team connectors to use the `LocalFileSystemConnector` regardless of their original configuration.
This is useful for testing and development without external dependencies.

## Usage Examples

### Basic Usage

```yaml
# config/model_config.yaml
model_name: "ibm-granite/granite-3.0-1b-a400m-instruct"
teams:
  - name: "team1"
    connector:
      type: "github"
      repo_owner: "${GITHUB_ORG}"
      repo_name: "repo1"
      token: "${GITHUB_TOKEN}"
```

### Environment Variables

```bash
# Set required environment variables
export GITHUB_ORG=my-org
export GITHUB_TOKEN=ghp_123456789abcdef

# Use local-only mode for testing
export GRANITE_LOCAL_ONLY=true

# Override integration config
export INTEGRATION_CONFIG_PATH=config/local_integration.yaml
export GRANITE_CONFIG_OVERRIDE_MODE=replace
```

### API Usage

```python
from src.utils.config_utils import resolve_env_vars, load_config_with_env_vars
from src.utils.constants import DEFAULT_HTTP_TIMEOUT

# Load configuration with environment variables
config = load_config_with_env_vars("config/model_config.yaml")

# Use constants instead of hardcoded values
timeout = config.get("timeout", DEFAULT_HTTP_TIMEOUT)
```
