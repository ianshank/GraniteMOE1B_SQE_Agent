# Granite Test Case Generator

An AI-powered test case generation system leveraging IBM Granite Mixture of Experts (MoE) models with Retrieval-Augmented Generation (RAG) and Cache-Augmented Generation (CAG) capabilities.

## Features

- **AI-Powered Test Generation**: Utilizes IBM Granite 3.0 MoE models for intelligent test case creation
- **Multi-Team Support**: Orchestrates test generation across multiple teams with different workflows
- **Flexible Integrations**: Built-in connectors for Jira and GitHub issue tracking systems
- **RAG System**: Retrieval-Augmented Generation for context-aware test case creation
- **CAG Cache**: Cache-Augmented Generation for optimized performance and pattern reuse
- **MLX Optimization**: Apple Silicon optimization using MLX framework for M-series Macs
- **Graceful Fallbacks**: Robust error handling with BM25 fallback when embeddings unavailable
- **Comprehensive Testing**: 97% test coverage with unit, integration, and contract tests
- **Production Logging**: Structured logging throughout for monitoring and debugging

## Requirements

- Python 3.10+
- Conda (recommended) or pip
- Apple Silicon Mac (for MLX optimization) or x86_64 architecture
- 8GB+ RAM recommended
- API credentials for Jira/GitHub (if using integrations)

## Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/granite-test-generator.git
cd granite-test-generator

# Create conda environment
conda create -n granite-moe python=3.10
conda activate granite-moe

# Install PyTorch
conda install pytorch torchvision torchaudio -c pytorch

# Install dependencies
pip install -r requirements.txt
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/granite-test-generator.git
cd granite-test-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
granite-test-generator/
├── config/                 # Configuration files
│   ├── model_config.yaml  # Model settings
│   ├── training_config.yaml # Training parameters
│   └── integration_config.yaml # Integration settings
├── src/
│   ├── models/            # AI model implementations
│   │   ├── granite_moe.py # Granite MoE trainer
│   │   └── test_case_schemas.py # Test case data models
│   ├── data/              # Data processing modules
│   │   ├── rag_retriever.py # RAG implementation
│   │   ├── cag_cache.py   # CAG cache system
│   │   └── data_processors.py # Data transformation
│   ├── agents/            # AI agents
│   │   └── generation_agent.py # Test generation logic + suite helpers
│   ├── integration/       # External integrations
│   │   ├── team_connectors.py # Jira/GitHub connectors
│   │   └── workflow_orchestrator.py # Multi-team orchestration
│   └── utils/             # Utility modules
│       ├── chunking.py    # Document chunking
│       ├── kv_cache.py    # Key-value cache
│       └── debugging.py   # Debug utilities
├── data/                  # Data directories
│   ├── requirements/      # Requirements documents
│   ├── training/         # Training data
│   └── user_stories/     # User story examples
├── tests/                # Test suite
│   ├── unit/            # Unit tests
│   ├── contract/        # Contract tests
│   └── integration/     # Integration tests
├── notebooks/           # Jupyter notebooks
└── output/             # Generated test cases

```

## Quick Start

### 1. Configure Your Environment

**Option A: Using .env file (Recommended for Development)**

1. Copy the environment template:
```bash
cp .env.template .env
```

2. Edit `.env` with your actual values:
```bash
GITHUB_TOKEN=your-github-token
JIRA_API_TOKEN=your-jira-api-token
GITHUB_ORG=your-organization
JIRA_BASE_URL=https://your-instance.atlassian.net
```

3. Configuration will automatically use environment variables:
```yaml
# config/model_config.yaml
model_name: "ibm-granite/granite-3.0-1b-a400m-instruct"
teams:
  - name: "backend-team"
    connector:
      type: "jira"
      base_url: "${JIRA_BASE_URL}"
      username: "${JIRA_USERNAME}"
      api_token: "${JIRA_API_TOKEN}"
      project_key: "BACKEND"
    rag_enabled: true
    cag_enabled: true
    auto_push: false
```

**Option B: Direct Environment Variables (Production)**

```bash
export JIRA_API_TOKEN="your-jira-api-token"
export GITHUB_TOKEN="your-github-token"
export GITHUB_ORG="your-organization"
```

### 2. Run Test Case Generation

```python
# Run the complete pipeline
python main.py

# Or use the run_tests.py script for testing
python run_tests.py --coverage
```

## Usage Examples

### Generate Test Cases for a Single Team

```python
from src.models.granite_moe import GraniteMoETrainer
from src.agents.generation_agent import TestGenerationAgent
from src.data.rag_retriever import RAGRetriever
from src.data.cag_cache import CAGCache
from src.utils.kv_cache import KVCache

# Initialize components
trainer = GraniteMoETrainer()
rag = RAGRetriever()
cache = CAGCache(KVCache())
agent = TestGenerationAgent(trainer, rag, cache)

# Generate test cases
requirements = ["User should be able to login with valid credentials"]
test_cases = await agent.generate_test_cases_for_team("backend", requirements)
```

### Generate a Named Test Suite (Regression/E2E)

```python
from src.agents.generation_agent import TestGenerationAgent

# Given an initialized agent and a list of requirements
suite = await agent.generate_test_suite_for_team(
    team_name="backend",
    requirements=["# Login\nUser can login", "# Logout\nUser can logout"],
    suite_name="Regression Suite",
    description="Core regressions for auth"
)
print(suite.name)          # "Regression Suite"
print(len(suite.test_cases))  # equals number of requirements
```

### Use with Jira Integration

```python
from src.integration.team_connectors import JiraConnector
from src.integration.workflow_orchestrator import WorkflowOrchestrator, TeamConfiguration

# Setup Jira connector
jira = JiraConnector(
    base_url="https://your-instance.atlassian.net",
    username="your-email@example.com",
    api_token="your-token",
    project_key="PROJ"
)

# Configure team
config = TeamConfiguration(
    team_name="backend",
    connector=jira,
    auto_push=True
)

# Run orchestration
orchestrator = WorkflowOrchestrator(agent)
orchestrator.register_team(config)
results = await orchestrator.process_all_teams()
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific directories
pytest tests/unit/ -v
pytest tests/contract/ -v
pytest tests/integration/ -v

# Use markers (registered in pytest.ini)
pytest -m "regression" -v
pytest -m "e2e" -v
pytest -m "integration" -v
pytest -m "contract" -v

# Run tests for specific module
pytest tests/unit/test_workflow_orchestrator.py -v
```

## Performance Optimization

### Apple Silicon (M1/M2/M3) Optimization

The system automatically detects Apple Silicon and uses MLX for optimized inference:

```python
# Automatically handled in GraniteMoETrainer
trainer = GraniteMoETrainer()  # Uses MLX on Apple Silicon
```

### Caching Strategy

- **KVCache**: Persistent disk-based caching with LRU eviction
- **CAGCache**: In-memory pattern caching for frequent test patterns
- **ChromaDB**: Vector storage for RAG document persistence

## Configuration

> **📖 For detailed configuration options, environment variables, and best practices, see [Configuration Guide](docs/CONFIGURATION.md)**

### Model Configuration (`config/model_config.yaml`)

```yaml
model_name: "ibm-granite/granite-3.0-1b-a400m-instruct"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
max_length: 512
temperature: 0.7
```

### Training Configuration (`config/training_config.yaml`)

```yaml
num_epochs: 3
batch_size: 4
learning_rate: 5e-5
warmup_steps: 100
output_dir: "./models/fine_tuned_granite"
```

## Logging

The system uses structured logging with different levels:

- **INFO**: Major operations and milestones
- **DEBUG**: Detailed operation steps
- **WARNING**: Non-critical issues
- **ERROR**: Exceptions with stack traces

Configure logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 src/

# Run type checking
mypy src/

# Format code
black src/ tests/
```

## Roadmap

- [ ] Support for additional LLM models
- [ ] Web UI for test case generation
- [ ] Integration with more issue tracking systems
- [ ] Enhanced fine-tuning capabilities
- [ ] Distributed processing support
- [ ] REST API endpoint

---

**Note**: This project requires appropriate API credentials for external integrations. Ensure you have the necessary permissions before using Jira or GitHub connectors.
