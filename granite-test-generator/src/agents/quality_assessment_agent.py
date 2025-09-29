"""Dynamic test case generation utilities.

This module intentionally focuses solely on producing structured test plans and
cases from lightweight story and development plan inputs. No scoring or
third-party export responsibilities are retained so the agent remains focused on
synthesising actionable QA collateral.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from src.models.test_case_schemas import (
    TestCase,
    TestCasePriority,
    TestCaseType,
    TestStep,
    TestSuite,
)


@dataclass(slots=True)
class GenerationContext:
    """Context detected from the incoming story/dev plan inputs."""

    key: str
    label: str
    case_prefix: str
    default_type: TestCaseType
    default_priority: TestCasePriority
    frameworks: Dict[str, str] = field(default_factory=dict)
    focus_areas: List[str] = field(default_factory=list)


@dataclass(slots=True)
class TestPlanSpec:
    """Structured artefact returned by the generation agent."""

    story_breakdown: str
    dev_plan: str
    plan_summary: str
    suites: List[TestSuite]
    coverage_focus: Dict[str, float]
    generated_cases: List[TestCase]
    heuristics: Dict[str, Iterable[str]]


class QualityAssessmentAgent:
    """Agent dedicated to dynamically generating test collateral."""

    def __init__(self, coverage_target: float = 0.85) -> None:
        if not 0 < coverage_target <= 1:
            raise ValueError("coverage_target must be between 0 and 1")
        self.coverage_target = coverage_target

    def generate_test_specification(
        self, story_breakdown: str, dev_plan: str
    ) -> TestPlanSpec:
        context = self._detect_context(story_breakdown, dev_plan)
        stories = self._extract_stories(story_breakdown)
        cases = [self._build_case(story, idx, context) for idx, story in enumerate(stories, start=1)]
        suites = self._group_cases_into_suites(cases)
        coverage = self._compute_coverage_metrics(cases, stories)
        plan_summary = self._compose_plan_summary(context, stories, dev_plan)
        heuristics = {
            "frameworks": context.frameworks.values(),
            "focus_areas": context.focus_areas,
            "keywords_detected": self._detect_keywords(story_breakdown, dev_plan),
        }
        return TestPlanSpec(
            story_breakdown=story_breakdown,
            dev_plan=dev_plan,
            plan_summary=plan_summary,
            suites=suites,
            coverage_focus=coverage,
            generated_cases=cases,
            heuristics=heuristics,
        )

    def _detect_context(self, story_breakdown: str, dev_plan: str) -> GenerationContext:
        combined = f"{story_breakdown} {dev_plan}".lower()
        if any(keyword in combined for keyword in ("fastapi", "python", "pydantic")):
            return GenerationContext(
                key="python_service",
                label="FastAPI service",
                case_prefix="API",
                default_type=TestCaseType.INTEGRATION,
                default_priority=TestCasePriority.HIGH,
                frameworks={"unit": "pytest", "integration": "httpx"},
                focus_areas=["authentication", "data validation", "error handling"],
            )
        if any(keyword in combined for keyword in ("react", "next.js", "frontend", "typescript")):
            return GenerationContext(
                key="frontend_ui",
                label="React frontend",
                case_prefix="UI",
                default_type=TestCaseType.FUNCTIONAL,
                default_priority=TestCasePriority.MEDIUM,
                frameworks={"component": "jest", "e2e": "playwright"},
                focus_areas=["component state", "accessibility", "responsive layout"],
            )
        return GenerationContext(
            key="node_service",
            label="Node service",
            case_prefix="SRV",
            default_type=TestCaseType.INTEGRATION,
            default_priority=TestCasePriority.MEDIUM,
            frameworks={"unit": "jest", "integration": "supertest"},
            focus_areas=["rest endpoints", "database interaction", "logging"],
        )

    def _extract_stories(self, story_breakdown: str) -> List[str]:
        raw_lines = [line.strip(" -*\t") for line in story_breakdown.splitlines() if line.strip()]
        if raw_lines:
            return raw_lines
        return [story_breakdown.strip()] if story_breakdown.strip() else ["Unnamed scenario"]

    def _build_case(self, story: str, index: int, context: GenerationContext) -> TestCase:
        case_id = f"{context.case_prefix}-{index:03d}"
        priority = self._derive_priority(story, context)
        test_type = self._derive_type(story, context)
        preconditions = self._derive_preconditions(story, context)
        steps = self._derive_steps(story, context, index)
        expected = self._derive_expected_result(story, context)
        requirement_id = f"REQ-{index:03d}"
        return TestCase(
            id=case_id,
            summary=story,
            priority=priority,
            test_type=test_type,
            preconditions=preconditions,
            input_data={"scenario_index": index},
            steps=steps,
            expected_results=expected,
            requirements_traced=[requirement_id],
        )

    def _derive_priority(self, story: str, context: GenerationContext) -> TestCasePriority:
        lowered = story.lower()
        if any(keyword in lowered for keyword in ("security", "payment", "authentication")):
            return TestCasePriority.HIGH
        if "performance" in lowered:
            return TestCasePriority.MEDIUM
        return context.default_priority

    def _derive_type(self, story: str, context: GenerationContext) -> TestCaseType:
        lowered = story.lower()
        if any(keyword in lowered for keyword in ("api", "endpoint", "service")):
            return TestCaseType.INTEGRATION
        if any(keyword in lowered for keyword in ("ui", "page", "component", "button")):
            return TestCaseType.FUNCTIONAL
        if "performance" in lowered:
            return TestCaseType.PERFORMANCE
        return context.default_type

    def _derive_preconditions(self, story: str, context: GenerationContext) -> List[str]:
        preconditions = ["Test environment provisioned", "Required data available"]
        if "authentication" in story.lower():
            preconditions.append("Valid credentials configured")
        if context.key == "frontend_ui":
            preconditions.append("Browser test profile initialised")
        return preconditions

    def _derive_steps(self, story: str, context: GenerationContext, index: int) -> List[TestStep]:
        steps: List[TestStep] = []
        if context.key == "frontend_ui":
            steps.append(TestStep(step_number=1, action="Render target component", expected_result="Component loads without errors"))
            steps.append(TestStep(step_number=2, action="Trigger primary user interaction", expected_result="UI updates to reflect story outcome"))
            steps.append(
                TestStep(
                    step_number=3,
                    action="Validate accessibility hints",
                    expected_result="Accessibility hints such as ARIA roles and labels are present",
                )
            )
        else:
            steps.append(TestStep(step_number=1, action="Prepare request payload", expected_result="Payload reflects story input requirements"))
            steps.append(TestStep(step_number=2, action="Invoke service behaviour", expected_result="Service responds with success status"))
            steps.append(TestStep(step_number=3, action="Assert data side effects", expected_result="Persistence layer reflects expected changes"))
        if "error" in story.lower():
            steps.append(
                TestStep(
                    step_number=len(steps) + 1,
                    action="Simulate failure trigger",
                    expected_result="System returns descriptive error details",
                )
            )
        steps.append(
            TestStep(
                step_number=len(steps) + 1,
                action="Record observations",
                expected_result=f"Scenario {index} execution evidence captured",
            )
        )
        return steps

    def _derive_expected_result(self, story: str, context: GenerationContext) -> str:
        if context.key == "frontend_ui":
            return "User interface reflects state changes and remains accessible"
        if "performance" in story.lower():
            return "Response times stay within documented SLOs"
        if "error" in story.lower():
            return "Service returns actionable error payload"
        return "Service processes request and returns success response"

    def _group_cases_into_suites(self, cases: List[TestCase]) -> List[TestSuite]:
        suites: List[TestSuite] = []
        by_type: Dict[TestCaseType, List[TestCase]] = {}
        for case in cases:
            by_type.setdefault(case.test_type, []).append(case)
        total_cases = float(len(cases)) if cases else 1.0
        for test_type, grouped_cases in by_type.items():
            allocation = len(grouped_cases) / total_cases
            suites.append(
                TestSuite(
                    name=f"{test_type.value.title()} scenarios",
                    description=f"Covers {test_type.value.replace('_', ' ')} behaviour",
                    test_cases=grouped_cases,
                    coverage_metrics={
                        "target": self.coverage_target,
                        "allocation": round(allocation, 2),
                    },
                )
            )
        return suites

    def _compute_coverage_metrics(self, cases: Sequence[TestCase], stories: Sequence[str]) -> Dict[str, float]:
        unique_requirements = {req for case in cases for req in case.requirements_traced}
        story_count = max(len(stories), 1)
        coverage_ratio = len(unique_requirements) / story_count
        return {
            "overall_target": self.coverage_target,
            "achieved_story_coverage": round(coverage_ratio, 2),
            "total_cases": float(len(cases)),
        }

    def _compose_plan_summary(
        self, context: GenerationContext, stories: Sequence[str], dev_plan: str
    ) -> str:
        story_overview = ", ".join(stories[:3])
        framework_notes = ", ".join(
            f"{category}: {tool}" for category, tool in context.frameworks.items()
        )
        return (
            f"Context: {context.label}\n"
            f"Detected stories: {len(stories)}\n"
            f"Highlighted frameworks: {framework_notes}\n"
            f"Development considerations: {dev_plan.strip()}\n"
            f"Story focus preview: {story_overview}"
        )

    def _detect_keywords(self, story_breakdown: str, dev_plan: str) -> List[str]:
        keywords = {
            "auth": ("auth", "login", "token"),
            "data": ("data", "database", "persistence"),
            "ui": ("ui", "frontend", "component"),
            "performance": ("performance", "latency", "throughput"),
            "error": ("error", "failure", "exception"),
        }
        combined = f"{story_breakdown} {dev_plan}".lower()
        detected: List[str] = []
        for label, tokens in keywords.items():
            if any(token in combined for token in tokens):
                detected.append(label)
        return detected
