# Contributing to Granite Test Generator

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code follows the style guidelines.
6. Issue that pull request!

## Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable.
2. Update the ARCHITECTURE.md if you've made structural changes.
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent.
4. The PR will be merged once you have the sign-off of at least one maintainer.

## Any contributions you make will be under the MIT Software License

## Report bugs using GitHub's [issue tracker](https://github.com/yourusername/granite-test-generator/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/granite-test-generator/issues/new).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/granite-test-generator.git
   cd granite-test-generator
   ```

2. **Create a development environment**
   ```bash
   conda create -n granite-dev python=3.10
   conda activate granite-dev
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Code Style Guidelines

### Python Style

We follow PEP 8 with some modifications:

- Line length: 100 characters (not 79)
- Use type hints for all function signatures
- Use docstrings for all public functions/classes
- Prefer f-strings over `.format()` or `%` formatting

### Example Code Style

```python
from typing import List, Dict, Optional


class ExampleClass:
    """Brief description of the class.
    
    Longer description if needed, explaining the purpose
    and usage of the class.
    
    Attributes:
        attribute_name: Description of the attribute
    """
    
    def __init__(self, param: str) -> None:
        """Initialize the ExampleClass.
        
        Args:
            param: Description of the parameter
        """
        self.attribute_name = param
    
    def example_method(self, items: List[str]) -> Dict[str, int]:
        """Brief description of what the method does.
        
        Args:
            items: List of items to process
            
        Returns:
            Dictionary mapping items to their counts
            
        Raises:
            ValueError: If items is empty
        """
        if not items:
            raise ValueError("Items list cannot be empty")
            
        return {item: len(item) for item in items}
```

### Import Order

1. Standard library imports
2. Related third party imports
3. Local application/library specific imports

Each group should be separated by a blank line.

```python
import os
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
from transformers import AutoModel

from src.models.granite_moe import GraniteMoETrainer
from src.utils.logging import setup_logger
```

## Testing Guidelines

### Writing Tests

- Every new feature must include tests
- Every bug fix must include a test that reproduces the bug
- Aim for at least 80% code coverage
- Use descriptive test names that explain what is being tested

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

from src.module import function_to_test


class TestFunctionName:
    """Tests for function_name."""
    
    def test_normal_operation(self):
        """Test function with normal inputs."""
        result = function_to_test("input")
        assert result == "expected"
    
    def test_edge_case(self):
        """Test function with edge case inputs."""
        with pytest.raises(ValueError):
            function_to_test("")
    
    @patch('src.module.external_dependency')
    def test_with_mock(self, mock_dep):
        """Test function with mocked dependencies."""
        mock_dep.return_value = "mocked"
        result = function_to_test("input")
        assert result == "expected_with_mock"
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_specific.py

# Run with verbose output
pytest tests/ -v

# Run tests matching pattern
pytest tests/ -k "test_edge"
```

## Documentation Guidelines

### Docstring Format

We use Google-style docstrings:

```python
def function_name(param1: str, param2: int = 0) -> bool:
    """Brief description of function.
    
    Longer description if needed, explaining in detail what
    the function does and any important notes.
    
    Args:
        param1: Description of param1
        param2: Description of param2. Defaults to 0.
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not an integer
        
    Example:
        >>> function_name("test", 5)
        True
    """
    pass
```

### README Updates

When adding new features, update the README.md:

- Add to the Features section if it's a major feature
- Update the Usage Examples with a code example
- Update the Configuration section if new config options are added
- Add to the Roadmap section if it completes a planned feature

## Logging Guidelines

Use structured logging with appropriate levels:

```python
import logging

logger = logging.getLogger(__name__)

# Debug: Detailed information for diagnosing problems
logger.debug(f"Processing item {item_id} with options {options}")

# Info: General informational messages
logger.info(f"Started processing batch of {len(items)} items")

# Warning: Something unexpected but not critical
logger.warning(f"Retry attempt {attempt} for item {item_id}")

# Error: A serious problem occurred
logger.error(f"Failed to process item {item_id}: {error}", exc_info=True)
```

## Commit Message Guidelines

We follow the Conventional Commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that don't affect code meaning (formatting, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

### Examples

```
feat(rag): add ChromaDB persistence for vector storage

Implement persistent storage using ChromaDB to maintain vector
embeddings across sessions. This improves startup time and
reduces memory usage.

Closes #123
```

```
fix(connectors): handle timeout errors in Jira connector

Add proper exception handling for requests timeout errors.
Previous implementation would crash on network issues.

Fixes #456
```

