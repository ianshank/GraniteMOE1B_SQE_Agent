from __future__ import annotations

import pytest

from src.rag.models import CodeLanguage, CodePattern, CodeContext, CodeSnippet
from src.agents.prompt.enhanced_prompt_builder import build_test_generation_prompt


def test_code_context_validation():
    ctx = CodeContext(
        requirement="Generate API tests",
        language=CodeLanguage.python,
        patterns_needed=[CodePattern.api_endpoint, CodePattern.error_handling],
        dependencies=["fastapi"],
    )
    assert ctx.requirement.startswith("Generate")
    assert ctx.language == CodeLanguage.python
    assert CodePattern.api_endpoint in ctx.patterns_needed


def test_code_context_invalid_requirement():
    with pytest.raises(ValueError):
        CodeContext(requirement="  ", language=CodeLanguage.python)


def test_code_snippet_validation_and_scores():
    s = CodeSnippet(
        content="def foo():\n    return 1\n",
        language=CodeLanguage.python,
        pattern_type=CodePattern.function_definition,
        description="Simple function",
        quality_score=0.5,
        complexity_score=0.1,
        reusability_score=0.8,
    )
    assert s.quality_score == pytest.approx(0.5)
    assert s.reusability_score >= 0.0

    with pytest.raises(ValueError):
        CodeSnippet(
            content="",
            language=CodeLanguage.python,
            pattern_type=CodePattern.function_definition,
            description="x",
        )

    with pytest.raises(ValueError):
        CodeSnippet(
            content="x",
            language=CodeLanguage.python,
            pattern_type=CodePattern.function_definition,
            description="x",
            quality_score=2.0,
        )


def test_prompt_builder_includes_snippets_and_fences():
    snippet = CodeSnippet(
        content="def api_get():\n    pass\n",
        language=CodeLanguage.python,
        pattern_type=CodePattern.api_endpoint,
        description="API endpoint",
        quality_score=0.7,
        complexity_score=0.2,
        reusability_score=0.6,
        file_path="/tmp/app.py",
        line_start=10,
        line_end=15,
    )
    prompt = build_test_generation_prompt(
        requirement="Create tests for GET endpoint",
        language=CodeLanguage.python,
        code_snippets=[snippet],
    )
    assert "REQUIREMENT:" in prompt
    assert "RELEVANT CODE PATTERNS:" in prompt
    assert "```python" in prompt
    assert "def api_get():" in prompt


