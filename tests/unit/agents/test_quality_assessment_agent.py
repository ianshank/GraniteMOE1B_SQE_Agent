"""Unit tests for the dynamic QualityAssessmentAgent generator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.quality_assessment_agent import QualityAssessmentAgent


def test_generate_specification_for_python_service() -> None:
    agent = QualityAssessmentAgent(coverage_target=0.9)
    stories = """
    As an authenticated user I can create an order
    As an operator I can retrieve order history without errors
    """.strip()

    spec = agent.generate_test_specification(stories, "Implement FastAPI endpoints with SQLModel")

    assert "FastAPI service" in spec.plan_summary
    assert len(spec.generated_cases) == 2
    assert all(case.id.startswith("API-") for case in spec.generated_cases)
    assert pytest.approx(0.9, rel=1e-3) == spec.coverage_focus["overall_target"]
    assert spec.coverage_focus["achieved_story_coverage"] == pytest.approx(1.0)
    assert any("integration" in suite.name.lower() for suite in spec.suites)


def test_generate_specification_for_frontend_highlights_ui_flows() -> None:
    agent = QualityAssessmentAgent()
    stories = "As a shopper I interact with the product grid UI"

    spec = agent.generate_test_specification(stories, "React + TypeScript client with accessibility focus")

    case = spec.generated_cases[0]
    assert case.id.startswith("UI-")
    assert case.test_type.value == "functional"
    assert any("accessibility" in step.expected_result.lower() for step in case.steps)
    assert "React frontend" in spec.plan_summary
    assert "component" in ", ".join(spec.heuristics["focus_areas"])


@pytest.mark.contract
def test_suite_allocation_ratios_sum_to_one() -> None:
    agent = QualityAssessmentAgent()
    stories = """
    As a developer I monitor API errors for latency issues
    As a developer I investigate authentication failures
    """.strip()

    spec = agent.generate_test_specification(stories, "Node service with observability hooks")

    total_allocation = sum(suite.coverage_metrics["allocation"] for suite in spec.suites)
    assert pytest.approx(1.0, rel=1e-6) == total_allocation
    assert spec.heuristics["keywords_detected"]
    assert spec.coverage_focus["total_cases"] == float(len(spec.generated_cases))
