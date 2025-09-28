import pytest
from types import SimpleNamespace

from src.integration.workflow_orchestrator import WorkflowOrchestrator, TeamConfiguration


class Conn:
    def __init__(self, n=1):
        self.n = n

    def fetch_requirements(self):
        return [{"id": f"R{i}", "summary": "S", "description": ""} for i in range(self.n)]

    def push_test_cases(self, tcs):  # pragma: no cover - not used here
        return True


@pytest.mark.asyncio
async def test_flags_and_counts_variety():
    class Agent:
        async def generate_test_cases_for_team(self, team, reqs):
            return [SimpleNamespace(
                id=f"{team}-1", summary="s", steps=[], expected_results="",
                priority=SimpleNamespace(value="medium"),
                test_type=SimpleNamespace(value="functional"),
                requirements_traced=[],
            )]

    orch = WorkflowOrchestrator(Agent())
    orch.register_team(TeamConfiguration("a", Conn(2), rag_enabled=False, cag_enabled=False, auto_push=False))
    orch.register_team(TeamConfiguration("b", Conn(1), rag_enabled=True, cag_enabled=False, auto_push=False))
    orch.register_team(TeamConfiguration("c", Conn(3), rag_enabled=False, cag_enabled=True, auto_push=False))

    res = await orch.process_all_teams()
    assert set(res.keys()) == {"a", "b", "c"}
    assert all(isinstance(v, list) for v in res.values())

