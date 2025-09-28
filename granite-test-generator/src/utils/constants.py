"""
Centralized constants for the Granite Test Generator application.

This module contains constants that were previously hardcoded throughout the codebase.
Centralizing them here makes the code more maintainable and easier to configure.
"""

# HTTP and API related constants
DEFAULT_HTTP_TIMEOUT = 30  # seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.5
DEFAULT_BATCH_SIZE = 4

# GitHub API related constants
GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_DEFAULT_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

# Jira API related constants
JIRA_DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# File and path related constants
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_REQUIREMENTS_DIR = "data/requirements"
DEFAULT_USER_STORIES_DIR = "data/user_stories"
DEFAULT_CACHE_DIR = "cache"

# Model related constants
DEFAULT_MODEL_NAME = "ibm-granite/granite-3.0-1b-a400m-instruct"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 5
DEFAULT_MAX_TOKENS = 512

# Cache related constants
DEFAULT_CACHE_SIZE = 10000

# Environment variable names
ENV_INTEGRATION_CONFIG_PATH = "INTEGRATION_CONFIG_PATH"
ENV_LOCAL_ONLY_MODE = "GRANITE_LOCAL_ONLY"
ENV_CONFIG_OVERRIDE_MODE = "GRANITE_CONFIG_OVERRIDE_MODE"

# Configuration modes
CONFIG_MODE_MERGE = "merge"
CONFIG_MODE_REPLACE = "replace"

# Test case related constants
DEFAULT_TEST_CASE_PRIORITY = "medium"
