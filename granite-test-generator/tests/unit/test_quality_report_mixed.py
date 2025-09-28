import pytest
from types import SimpleNamespace

from src.integration.workflow_orchestrator import WorkflowOrchestrator, TeamConfiguration
from src.models.test_case_schemas import TestCase, TestStep, TestCasePriority, TestCaseType


def _tc(team: str, steps_n: int, prio: TestCasePriority, ttype: TestCaseType) -> TestCase:
    steps = [TestStep(step_number=i + 1, action=f"A{i}", expected_result="R") for i in range(steps_n)]
    return TestCase(
        id=f"{team}-{steps_n}",
        summary="s",
        priority=prio,
        test_type=ttype,
        steps=steps,
        expected_results="ok",
        team_context=team,
    )


def test_quality_report_weighted_and_totals():
    orch = WorkflowOrchestrator(SimpleNamespace())

    # Register teams to count as processed
    orch.register_team(TeamConfiguration("t1", SimpleNamespace()))
    orch.register_team(TeamConfiguration("t2", SimpleNamespace()))
    orch.register_team(TeamConfiguration("t3", SimpleNamespace()))

    # Populate results with mixed sizes/priorities/types and steps
    orch.results_cache = {
        "t1": [_tc("t1", 1, TestCasePriority.HIGH, TestCaseType.FUNCTIONAL)],
        "t2": [
            _tc("t2", 2, TestCasePriority.MEDIUM, TestCaseType.INTEGRATION),
            _tc("t2", 3, TestCasePriority.MEDIUM, TestCaseType.INTEGRATION),
        ],
        "t3": [
            _tc("t3", 4, TestCasePriority.LOW, TestCaseType.UNIT),
        ],
    }

    report = orch.generate_quality_report()

    # Totals
    assert report["total_test_cases"] == 4
    assert report["teams_processed"] == 3
    assert report["teams_with_results"] == 3

    tm = report["team_metrics"]
    # Per-team averages and totals
    assert tm["t1"]["test_case_count"] == 1 and tm["t1"]["total_steps"] == 1 and tm["t1"]["average_steps_per_test"] == 1.0
    assert tm["t2"]["test_case_count"] == 2 and tm["t2"]["total_steps"] == 5 and tm["t2"]["average_steps_per_test"] == 2.5
    assert tm["t3"]["test_case_count"] == 1 and tm["t3"]["total_steps"] == 4 and tm["t3"]["average_steps_per_test"] == 4.0

    # Weighted average across teams: sum(total_steps)/sum(count)
    total_steps = sum(m["total_steps"] for m in tm.values())
    total_cases = sum(m["test_case_count"] for m in tm.values())
    assert total_steps == 1 + 5 + 4 == 10
    assert total_cases == 1 + 2 + 1 == 4
    weighted_avg = total_steps / total_cases
    assert weighted_avg == 2.5
