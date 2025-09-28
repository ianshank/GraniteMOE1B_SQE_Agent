from pydantic import BaseModel, Field
from typing import List, Dict, Any

class TestCase(BaseModel):
    summary: str
    input_data: Dict[str, Any]
    steps: List[Dict[str, Any]]
    expected_results: str
    requirements_traced: List[str] = Field(default_factory=list)
    quality_score: float = 0.0

class QualityAssessmentAgent:
    """Agent for automated evaluation of software test case quality."""

    def __init__(self, scoring_rules: Dict = None):
        self.scoring_rules = scoring_rules or {
            "summary": 1,
            "input_data": 1,
            "steps": 2,
            "expected_results": 1,
            "coverage": 2
        }

    def assess_quality(self, test_case: TestCase, requirements: List[str]) -> Dict[str, Any]:
        score = 0
        missing_sections = []

        if not test_case.summary:
            missing_sections.append("summary")
        else:
            score += self.scoring_rules["summary"]

        if not test_case.input_data:
            missing_sections.append("input_data")
        else:
            score += self.scoring_rules["input_data"]

        if not test_case.steps or len(test_case.steps) < 2:
            missing_sections.append("steps")
        else:
            score += self.scoring_rules["steps"]

        if not test_case.expected_results:
            missing_sections.append("expected_results")
        else:
            score += self.scoring_rules["expected_results"]

        coverage = 0
        if test_case.requirements_traced:
            coverage = len(set(test_case.requirements_traced) & set(requirements)) / max(1, len(requirements))
            score += self.scoring_rules["coverage"] * coverage

        result = {
            "score": score,
            "missing": missing_sections,
            "coverage": coverage,
            "passed": len(missing_sections) == 0
        }
        return result

    def assess_suite(self, suite: List[TestCase], requirements: List[str]) -> List[Dict[str, Any]]:
        return [self.assess_quality(tc, requirements) for tc in suite]