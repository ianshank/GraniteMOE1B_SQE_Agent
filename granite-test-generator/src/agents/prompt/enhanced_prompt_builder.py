"""
Prompt builder for code-aware test generation.

Pure functions only. No side effects or external dependencies. This module is
safe to import in any environment and does not change existing behavior unless
explicitly called by the agent logic.
"""

from __future__ import annotations

import logging
from typing import Iterable, List

from src.rag.models import CodeSnippet, CodeLanguage


logger = logging.getLogger(__name__)


def build_test_generation_prompt(
    requirement: str,
    language: CodeLanguage,
    code_snippets: Iterable[CodeSnippet] | None = None,
    extra_instructions: List[str] | None = None,
) -> str:
    """Construct a test generation prompt optionally enriched with code snippets.

    Args:
        requirement: Human-readable requirement or story to generate tests for.
        language: Target language for code examples fenced in the prompt.
        code_snippets: Optional iterable of `CodeSnippet` providing relevant
            patterns from the codebase.
        extra_instructions: Optional additional bullet-point guidance.

    Returns:
        A multi-line string suitable to pass to a generation agent/model.
    """
    logger.debug(
        "Building test generation prompt (req_len=%d, lang=%s, snippets=%s)",
        len(requirement or ""), language.value, "yes" if code_snippets else "no",
    )

    parts: List[str] = [
        f"REQUIREMENT: {requirement}",
        f"TARGET LANGUAGE: {language.value}",
        "INSTRUCTIONS:",
        "- Use relevant code patterns as guidance",
        "- Include proper error handling and logging",
        "- Add concise docs and follow project conventions",
        "- Prefer deterministic tests with clear assertions",
    ]

    if extra_instructions:
        parts.extend(extra_instructions)

    if code_snippets:
        parts.append("\nRELEVANT CODE PATTERNS:")
        for i, s in enumerate(list(code_snippets)[:5]):
            parts.append(f"\nExample {i+1} - {s.description}")
            parts.append(
                f"Pattern: {s.pattern_type.value} | Quality: {s.quality_score:.2f}"
            )
            file_loc = (
                f"{s.file_path}:{s.line_start}-{s.line_end}" if s.file_path else "Unknown"
            )
            parts.append(f"File: {file_loc}")
            parts.append(f"```{s.language.value}")
            parts.append(s.content)
            parts.append("```")

    prompt = "\n".join(parts)
    logger.debug("Prompt built with length=%d chars", len(prompt))
    return prompt


