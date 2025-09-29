"""
Typed models for code-aware RAG features.

These models are intentionally minimal and framework-agnostic to enable
reuse across indexing, retrieval, and prompt construction without creating
new runtime dependencies for existing flows.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, validator


logger = logging.getLogger(__name__)


class CodeLanguage(str, Enum):
    python = "python"
    javascript = "javascript"
    typescript = "typescript"
    java = "java"
    csharp = "csharp"
    cpp = "cpp"
    go = "go"
    rust = "rust"
    sql = "sql"
    html = "html"
    css = "css"


class CodePattern(str, Enum):
    class_definition = "class_definition"
    function_definition = "function_definition"
    api_endpoint = "api_endpoint"
    database_query = "database_query"
    algorithm = "algorithm"
    design_pattern = "design_pattern"
    error_handling = "error_handling"
    testing_pattern = "testing_pattern"
    authentication = "authentication"
    configuration = "configuration"


class CodeContext(BaseModel):
    """Context describing a code generation or retrieval request.

    This mirrors the information needed to build effective RAG queries
    without binding to a specific agent framework.
    """

    requirement: str
    language: CodeLanguage
    patterns_needed: list[CodePattern] = Field(default_factory=list)
    existing_codebase: Optional[str] = None
    dependencies: list[str] = Field(default_factory=list)
    architectural_style: Optional[str] = None
    performance_requirements: Optional[str] = None
    security_requirements: Optional[str] = None

    @validator("requirement")
    def _requirement_not_empty(cls, v: str) -> str:  # noqa: D401
        if not isinstance(v, str) or not v.strip():
            raise ValueError("requirement must be a non-empty string")
        return v


class CodeSnippet(BaseModel):
    """Represents a code snippet with rich metadata for ranking and display."""

    content: str
    language: CodeLanguage
    pattern_type: CodePattern
    description: str
    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    quality_score: float = 0.0
    complexity_score: float = 0.0
    reusability_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @validator("content")
    def _content_not_empty(cls, v: str) -> str:  # noqa: D401
        if not isinstance(v, str) or not v.strip():
            raise ValueError("content must be a non-empty string")
        return v

    @validator("quality_score", "complexity_score", "reusability_score")
    def _score_in_range(cls, v: float) -> float:  # noqa: D401
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError("scores must be in [0.0, 1.0]")
        return float(v)

    class Config:
        # Keep Enums as Enum instances to support .value in downstream consumers
        use_enum_values = False


