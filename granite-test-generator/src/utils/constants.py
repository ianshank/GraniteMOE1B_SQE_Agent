"""
Constants and default values for the Granite Test Generator.
Centralized location for hardcoded values to improve maintainability.
"""

from typing import List, Dict

# Model Defaults
DEFAULT_MODEL_NAME = "ibm-granite/granite-3.0-1b-a400m-instruct"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Training Defaults
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_WARMUP_STEPS = 100

# RAG Configuration
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 5

# Cache Configuration
DEFAULT_CACHE_SIZE = 10000

# Network Timeouts (seconds)
DEFAULT_HTTP_TIMEOUT = 30
DEFAULT_API_TIMEOUT = 30

# File System Defaults
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_MODELS_DIR = "./models/fine_tuned_granite"
DEFAULT_CACHE_DIR = "cache"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_DATA_DIR = "data"
DEFAULT_REQUIREMENTS_DIR = "data/requirements"
DEFAULT_TRAINING_DIR = "data/training"
DEFAULT_USER_STORIES_DIR = "data/user_stories"

# File Extensions
SUPPORTED_FILE_EXTENSIONS = [".md", ".txt", ".json", ".yaml", ".yml"]
DEFAULT_FILE_EXTENSIONS = [".md", ".txt", ".json"]

# GitHub API Configuration
GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_API_VERSION = "application/vnd.github.v3+json"

# Priority Mappings
PRIORITY_MAPPINGS = {
    "P1": "high",
    "P2": "medium", 
    "P3": "low"
}

# Magic Numbers for Processing
DEFAULT_MAX_LENGTH = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_LINE_LENGTH = 100

# Retry Configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1

# Logging Configuration
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Application Metadata
APP_NAME = "Granite Test Generator"
APP_VERSION = "3.0"
APP_DESCRIPTION = "MoE-powered agent for generating test cases from requirements"

# Template Generation Patterns
TEMPLATE_PATTERNS = {
    "login": [
        "Open login page -> Page loads",
        "Enter valid credentials -> Credentials accepted",
        "Submit form -> User redirected to dashboard",
        "Verify session cookie -> Session established"
    ],
    "upload": [
        "Prepare sample file -> File ready",
        "Upload file -> Server responds 202",
        "Poll status -> Processing completes",
        "Validate stored object -> MD5 matches"
    ],
    "default": [
        "Set up pre-conditions -> Environment ready",
        "Execute primary action -> Action succeeds",
        "Observe output -> Output matches requirement",
        "Clean up data -> State restored"
    ]
}

# Environment Variables
ENV_VARS = {
    "JIRA_API_TOKEN": "JIRA_API_TOKEN",
    "GITHUB_TOKEN": "GITHUB_TOKEN",
    "HF_TOKEN": "HF_TOKEN",
    "OPENAI_API_KEY": "OPENAI_API_KEY"
}