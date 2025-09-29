"""Integration-style tests for the dynamic QualityAssessmentAgent."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.quality_assessment_agent import QualityAssessmentAgent
from src.models.test_case_schemas import TestCaseType


def test_generate_plan_outputs_unique_case_identifiers() -> None:
    agent = QualityAssessmentAgent()
    stories = """
    As an API consumer I receive descriptive errors when validation fails
    As an API consumer I retry requests after transient failures
    As an operator I view performance dashboards
    """.strip()

    spec = agent.generate_test_specification(stories, "Node.js service emitting metrics")
    case_ids = {case.id for case in spec.generated_cases}

    assert len(case_ids) == len(spec.generated_cases)
    assert any(case.test_type is TestCaseType.PERFORMANCE for case in spec.generated_cases)
    assert any("error" in step.expected_result.lower() for case in spec.generated_cases for step in case.steps)


def test_plan_summary_reflects_detected_keywords_and_focus() -> None:
    agent = QualityAssessmentAgent(coverage_target=0.92)
    spec = agent.generate_test_specification(
        "As a user I authenticate and access protected pages",
        "React frontend using token based auth",
    )

    assert "React frontend" in spec.plan_summary
    assert "auth" in spec.heuristics["keywords_detected"]
    assert spec.coverage_focus["overall_target"] == pytest.approx(0.92)
    assert spec.suites[0].test_cases[0].preconditions
