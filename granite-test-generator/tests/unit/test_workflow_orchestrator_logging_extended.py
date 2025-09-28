import asyncio
import time
import logging
import pytest
from types import SimpleNamespace

from src.integration.workflow_orchestrator import WorkflowOrchestrator, TeamConfiguration


class _ConnectorOK:
    def __init__(self, req_count: int = 1):
        self.req_count = req_count
        self.push_called = False

    def fetch_requirements(self):
        return [{"id": f"R{i}", "summary": "S", "description": ""} for i in range(self.req_count)]

    def push_test_cases(self, test_cases):
        self.push_called = True
        return True


class _ConnectorFail:
    def fetch_requirements(self):
        raise Exception("boom")

    def push_test_cases(self, test_cases):  # pragma: no cover - not reached
        return False


@pytest.mark.asyncio
async def test_logging_and_parallelism_extended(caplog):
    caplog.set_level(logging.DEBUG)

    async def slow_generate(team, reqs):
        await asyncio.sleep(0.1)
        return [SimpleNamespace(
            id=f"{team}-1",
            summary="s",
            steps=[],
            expected_results="",
            priority=SimpleNamespace(value="medium"),
            test_type=SimpleNamespace(value="functional"),
            requirements_traced=[],
        )]

    class Agent:
        async def generate_test_cases_for_team(self, team, reqs):
            return await slow_generate(team, reqs)

    orch = WorkflowOrchestrator(Agent())
    # Register three teams: two ok (one auto_push) and one failing
    orch.register_team(TeamConfiguration("t1", _ConnectorOK(), auto_push=True))
    orch.register_team(TeamConfiguration("t2", _ConnectorOK()))
    orch.register_team(TeamConfiguration("t3", _ConnectorFail()))

    start = time.time()
    res = await orch.process_all_teams()
    elapsed = time.time() - start

    # Parallelism: three teams with 0.1s work should complete well under 0.3s
    assert elapsed < 0.25

    # Logging assertions for start/end and task creation
    messages = [r.message for r in caplog.records]
    assert any("Starting test case generation for 3 registered teams" in m for m in messages)
    assert any("Creating processing task for team: t1" in m for m in messages)
    assert any("Creating processing task for team: t2" in m for m in messages)
    assert any("Creating processing task for team: t3" in m for m in messages)
    assert any("Executing 3 team processing tasks in parallel" in m for m in messages)

    # Success and error logging
    assert any("Team 't1' processing completed with" in m for m in messages)
    assert any("Team 't2' processing completed with" in m for m in messages)
    assert any("Error processing team 't3'" in m for m in messages)
    # Orchestrator swallows per-team exceptions and returns empty lists, so
    # from the top-level perspective all teams are "successful" in gather()
    assert any("All teams processing completed. Successful: 3, Failed: 0" in m for m in messages)

    # Results and report
    assert set(res.keys()) == {"t1", "t2", "t3"}
    rep = orch.generate_quality_report()
    assert rep["report_status"] in {"no_data", "success"}
    log_msgs = [r.message for r in caplog.records]
    assert any("Generating quality report" in m for m in log_msgs)
    assert any("Quality report generated" in m for m in log_msgs)
