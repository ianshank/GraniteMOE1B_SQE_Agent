# Granite Test Generator - System Architecture

## Overview

The Granite Test Generator is a modular, AI-powered system designed to automatically generate test cases from requirements using IBM's Granite Mixture of Experts (MoE) models, enhanced with Retrieval-Augmented Generation (RAG) and Cache-Augmented Generation (CAG) capabilities.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           External Systems                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │     Jira     │  │    GitHub    │  │  Future APIs │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
└─────────┼──────────────────┼──────────────────┼─────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Integration Layer                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Team Connectors                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │   │
│  │  │JiraConnector │  │GitHubConnect│  │ BaseConnector│         │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘         │   │
│  └─────────────────────────────┬───────────────────────────────────┘   │
│                                │                                         │
│  ┌─────────────────────────────▼───────────────────────────────────┐   │
│  │                   Workflow Orchestrator                          │   │
│  │  • Team Registration  • Parallel Processing  • Quality Reports  │   │
│  └─────────────────────────────┬───────────────────────────────────┘   │
└────────────────────────────────┼─────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Agent Layer                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 Test Generation Agent                            │   │
│  │  • Requirement Analysis  • Test Case Creation  • Validation     │   │
│  └──────────┬──────────────────────┬──────────────────┬───────────┘   │
└─────────────┼──────────────────────┼──────────────────┼─────────────────┘
              │                      │                    │
              ▼                      ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         AI/ML Layer                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │  Granite MoE    │  │  RAG Retriever  │  │   CAG Cache     │        │
│  │    Trainer      │  │                 │  │                 │        │
│  │ • Fine-tuning   │  │ • FAISS Vector  │  │ • Pattern Cache │        │
│  │ • MLX Inference │  │ • BM25 Fallback │  │ • Team Caches   │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
              │                      │                    │
              ▼                      ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Storage Layer                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │   KV Cache      │  │   ChromaDB      │  │  File System    │        │
│  │ • LRU Eviction  │  │ • Vector Store  │  │ • Requirements  │        │
│  │ • Persistence   │  │ • Persistence   │  │ • Test Cases    │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Integration Layer

#### Team Connectors
- **Purpose**: Abstract interface for external issue tracking systems
- **Key Components**:
  - `TeamConnector` (ABC): Base interface defining `fetch_requirements()` and `push_test_cases()`
  - `JiraConnector`: Implements Jira REST API v3 integration
  - `GitHubConnector`: Implements GitHub Issues API integration
- **Features**:
  - Structured logging for all operations
  - Timeout handling (30s default)
  - Error handling with detailed exceptions
  - Automatic retry logic (planned)

#### Workflow Orchestrator
- **Purpose**: Coordinate test generation across multiple teams
- **Key Features**:
  - Asynchronous parallel processing using `asyncio`
  - Team configuration management
  - Quality metrics generation
  - Results caching
- **Processing Flow**:
  1. Fetch requirements from connectors
  2. Generate test cases via agent
  3. Add requirement traceability
  4. Optional auto-push to source system
  5. Cache results and generate reports

### 2. Agent Layer

#### Test Generation Agent
- **Purpose**: Orchestrate AI-powered test case generation
- **Key Responsibilities**:
  - Coordinate between AI models and data systems
  - Implement test generation strategies
  - Manage context retrieval and caching
- **Tools Integration**:
  - LangChain agents for tool orchestration
  - Custom tools for test case creation
  - Context-aware generation using RAG/CAG

### 3. AI/ML Layer

#### Granite MoE Trainer
- **Purpose**: Manage Granite model training and inference
- **Features**:
  - Automatic device detection (CUDA/MPS/CPU)
  - MLX optimization for Apple Silicon
  - Fine-tuning capabilities
  - Structured prompt templates
- **Model Support**:
  - IBM Granite 3.0 1B/3B models
  - Hugging Face Transformers integration
  - Quantization support (planned)

#### RAG Retriever
- **Purpose**: Provide context-aware retrieval for test generation
- **Architecture**:
  ```
  Documents → Chunking → Embeddings → Vector Store
                    ↓                       ↓
                 BM25 Index ←→ Ensemble Retriever
  ```
- **Fallback Strategy**:
  - Primary: FAISS vector search with embeddings
  - Fallback: BM25 keyword search
  - Graceful degradation on component failure

#### CAG Cache
- **Purpose**: Cache-augmented generation for performance
- **Features**:
  - In-memory pattern caching
  - Team-specific caches
  - Common pattern preloading
  - Integration with KV Cache for persistence

### 4. Storage Layer

#### KV Cache
- **Purpose**: Persistent key-value storage for embeddings and responses
- **Implementation**:
  - File-based storage using pickle
  - SHA256 content hashing
  - LRU eviction strategy
  - Metadata persistence in JSON
- **Configuration**:
  - Default max size: 15,000 entries
  - Automatic eviction on overflow
  - Tag-based retrieval support

#### ChromaDB Integration
- **Purpose**: Vector database for document persistence
- **Features**:
  - Persistent storage in `./chroma_db`
  - Collection-based organization
  - Metadata filtering
  - Fallback handling on failure

### 5. Data Processing

#### Document Chunking
- **Purpose**: Intelligent document splitting for retrieval
- **Strategy**:
  - Recursive character text splitting
  - Metadata preservation
  - Source type tracking
  - Team context association

#### Data Processors
- **Purpose**: Transform various input formats
- **Supported Formats**:
  - Markdown requirements
  - JSON user stories
  - CSV test cases
  - PDF documents (planned)

## Data Flow

```
Requirements Document
        │
        ▼
┌─────────────────┐
│   Chunking      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Embedding + Indexing        │
│  ┌─────────┐  ┌──────────┐ │
│  │ FAISS   │  │  BM25    │ │
│  └─────────┘  └──────────┘ │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────┐
│  RAG Retrieval  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Test Generation │
│     Agent       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Granite MoE    │
│     Model       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Test Cases     │
└─────────────────┘
```

## Error Handling Strategy

### Graceful Degradation
1. **Embeddings Unavailable**: Fall back to BM25 search
2. **FAISS Import Failure**: Use BM25 only
3. **ChromaDB Failure**: Continue without persistence
4. **Connector Failure**: Log error, return empty list
5. **Model Loading Failure**: Return cached results if available

### Logging Hierarchy
- **ERROR**: Component failures, exceptions
- **WARNING**: Degraded functionality, partial failures
- **INFO**: Major operations, milestones
- **DEBUG**: Detailed operations, data flow

## Security Considerations

1. **API Credentials**: Environment variables, never hardcoded
2. **Input Validation**: All external inputs validated
3. **Timeout Protection**: 30s default on all HTTP requests
4. **Error Messages**: Sanitized to prevent information leakage
5. **File Permissions**: Restricted access to cache directories

## Performance Optimizations

1. **Parallel Processing**: Async team processing
2. **Caching Strategy**: Multi-level caching (KV, CAG, ChromaDB)
3. **MLX Optimization**: Apple Silicon acceleration
4. **Batch Processing**: Efficient document indexing
5. **LRU Eviction**: Memory-bounded caching

## Extensibility Points

### Adding New Connectors
1. Implement `TeamConnector` interface
2. Add configuration schema
3. Register in orchestrator
4. Add contract tests

### Adding New Models
1. Extend `GraniteMoETrainer`
2. Add model configuration
3. Update prompt templates
4. Add model-specific tests

### Adding New Storage Backends
1. Implement storage interface
2. Add configuration options
3. Update retriever integration
4. Add persistence tests

## Testing Architecture

```
┌─────────────────────────────────────────┐
│              Test Suite                  │
├─────────────────┬───────────────────────┤
│   Unit Tests    │   97% Coverage        │
│   • Isolation   │   • Mocking           │
│   • Fast        │   • Edge Cases        │
├─────────────────┼───────────────────────┤
│ Contract Tests  │   API Validation      │
│   • Connectors  │   • Payload Checks    │
│   • Schemas     │   • Error Scenarios   │
├─────────────────┼───────────────────────┤
│Integration Tests│   E2E Workflows       │
│   • Full Flow   │   • Real Components   │
│   • Async Ops   │   • Performance       │
└─────────────────┴───────────────────────┘
```

## Deployment Considerations

### Resource Requirements
- **Memory**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for models + cache
- **CPU**: 4+ cores for parallel processing
- **GPU**: Optional, CUDA/MPS supported

### Configuration Management
- YAML-based configuration
- Environment variable overrides
- Per-team customization
- Runtime reconfiguration support

### Monitoring Integration
- Structured JSON logging
- Metrics export (planned)
- Health check endpoints (planned)
- Performance profiling hooks

