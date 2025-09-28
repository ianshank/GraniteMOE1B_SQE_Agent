import pytest
from unittest.mock import Mock

from src.integration.workflow_orchestrator import WorkflowOrchestrator, TeamConfiguration


class BoomAgent:
    async def generate_test_cases_for_team(self, team, requirements):  # pragma: no cover - exercised by orchestrator
        raise RuntimeError("agent boom")


class DummyConnector:
    def fetch_requirements(self):
        return [{"id": "R1", "summary": "s", "description": ""}]

    def push_test_cases(self, tcs):  # pragma: no cover - not invoked
        return True


@pytest.mark.asyncio
async def test_process_team_agent_exception_handled():
    orch = WorkflowOrchestrator(BoomAgent())
    orch.register_team(TeamConfiguration("t", DummyConnector()))
    results = await orch.process_all_teams()
    assert results["t"] == []

