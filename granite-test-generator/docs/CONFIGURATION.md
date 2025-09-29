# Configuration Guide

This document explains how to configure the Granite Test Generator system with externalized configuration values instead of hardcoded constants.

## Overview

The system has been updated to remove hardcoded values and use configurable parameters. Configuration is handled through:

1. **Constants file** (`src/utils/constants.py`) - Default values and system constants
2. **Environment variables** - Runtime configuration and secrets
3. **YAML configuration files** - Structure and team configurations
4. **Configuration utilities** - Environment variable substitution and validation

## Configuration Files

### Environment Variables (.env)

Create a `.env` file in the project root (copy from `.env.template`):

```bash
# Copy the template
cp .env.template .env

# Edit with your actual values
nano .env
```

**Important:** Never commit `.env` files with actual secrets to version control.

### Configuration Structure

#### 1. Constants (`src/utils/constants.py`)

Contains default values and system constants:

```python
# Model Defaults
DEFAULT_MODEL_NAME = "ibm-granite/granite-3.0-1b-a400m-instruct"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4

# Network Configuration
DEFAULT_HTTP_TIMEOUT = 30
GITHUB_API_BASE_URL = "https://api.github.com"

# File System Paths
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_MODELS_DIR = "./models/fine_tuned_granite"
```

#### 2. Model Configuration (`config/model_config.yaml`)

Now supports environment variable substitution:

```yaml
model_name: "ibm-granite/granite-3.0-1b-a400m-instruct"

teams:
  - name: "example_team"
    connector:
      type: "github"
      repo_owner: "${GITHUB_ORG}"           # From environment
      repo_name: "example-repo"
      token: "${GITHUB_TOKEN}"              # From environment
    rag_enabled: true
    cag_enabled: true
```

#### 3. Environment Variable Formats

The system supports these formats in YAML files:

- `${VAR_NAME}` - Required variable (error if not found)
- `${VAR_NAME:default}` - Optional with default value

## Environment Variables Reference

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GITHUB_TOKEN` | GitHub API token | `ghp_xxxxxxxxxxxx` |
| `JIRA_API_TOKEN` | Jira API token | `ATxxxxxxxxxxxxxxx` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GITHUB_ORG` | GitHub organization | `your-org` |
| `JIRA_BASE_URL` | Jira instance URL | `https://company.atlassian.net` |
| `DEFAULT_LOG_LEVEL` | Logging level | `INFO` |

### System Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OUTPUT_DIR` | Output directory | `output` |
| `MODELS_DIR` | Model storage path | `./models/fine_tuned_granite` |
| `CACHE_DIR` | Cache directory | `cache` |

## Usage Examples

### Basic Usage

```python
from src.main import GraniteTestCaseGenerator

# Loads config/model_config.yaml with environment substitution
generator = GraniteTestCaseGenerator()
await generator.initialize_system()
```

### Custom Configuration

```python
# Use custom config path
generator = GraniteTestCaseGenerator("path/to/custom/config.yaml")

# Or provide config directly
config = {
    "model_name": "custom-model",
    "teams": [...]
}
generator = GraniteTestCaseGenerator(config_dict=config)
```

### Environment Loading

```python
from src.utils.config_utils import load_env_file, get_config_with_defaults

# Load .env file manually
load_env_file('.env')

# Load config with environment substitution
config = get_config_with_defaults('config/model_config.yaml')
```

## Migration from Hardcoded Values

### Before (Hardcoded)

```python
# Old hardcoded values
timeout = 30
api_url = "https://api.github.com"
batch_size = 4
```

### After (Configurable)

```python
from src.utils.constants import DEFAULT_HTTP_TIMEOUT, GITHUB_API_BASE_URL, DEFAULT_BATCH_SIZE

# Use constants
timeout = DEFAULT_HTTP_TIMEOUT
api_url = GITHUB_API_BASE_URL
batch_size = DEFAULT_BATCH_SIZE
```

## Security Best Practices

1. **Never commit secrets** - Use `.env` files for local development
2. **Use environment variables** - For production deployments
3. **Validate configuration** - Check required variables at startup
4. **Rotate tokens regularly** - Update API tokens periodically
5. **Use least privilege** - Grant minimal required permissions

## Configuration Validation

The system validates configuration at startup:

```python
from src.utils.config_utils import validate_required_env_vars

# Check required variables
required = ['GITHUB_TOKEN', 'JIRA_API_TOKEN']
validate_required_env_vars(required)
```

## Troubleshooting

### Common Issues

1. **Missing environment variables**
   ```
   ValueError: Missing required environment variables: GITHUB_TOKEN
   ```
   **Solution:** Set the missing variables in your `.env` file

2. **Invalid YAML syntax**
   ```
   yaml.YAMLError: Invalid YAML syntax
   ```
   **Solution:** Check YAML indentation and syntax

3. **File not found errors**
   ```
   FileNotFoundError: Configuration file not found
   ```
   **Solution:** Ensure config file exists or provide correct path

### Debug Configuration

Enable debug logging to see configuration loading:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Development vs Production

### Development Setup

```bash
# Use .env file for local development
cp .env.template .env
# Edit .env with development values
```

### Production Deployment

```bash
# Set environment variables directly
export GITHUB_TOKEN="production-token"
export JIRA_API_TOKEN="production-token"
# Application will use environment variables instead of .env file
```

## Additional Resources

- [Environment Variables Best Practices](https://12factor.net/config)
- [YAML Specification](https://yaml.org/spec/)
- [Security Guidelines](../SECURITY.md)