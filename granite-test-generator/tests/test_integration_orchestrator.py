import pytest
from types import SimpleNamespace
import asyncio


@pytest.mark.integration
@pytest.mark.e2e
def test_orchestrator_e2e_minimal():
    # Lazy import to avoid heavy deps on collection
    from src.integration.workflow_orchestrator import WorkflowOrchestrator, TeamConfiguration

    class StubConnector:
        def fetch_requirements(self):
            return [{'id': 'R1', 'summary': 'Login works', 'description': ''}]

        def push_test_cases(self, test_cases):
            return True

    class StubAgent:
        async def generate_test_cases_for_team(self, team, requirements):
            # Minimal TestCase object
            return [SimpleNamespace(
                id='TC-001',
                summary='Case', steps=[], expected_results='',
                priority=SimpleNamespace(value='medium'),
                test_type=SimpleNamespace(value='functional'),
                requirements_traced=[]
            )]

    orch = WorkflowOrchestrator(StubAgent())
    orch.register_team(TeamConfiguration(team_name='teamA', connector=StubConnector(), auto_push=True))

    results = asyncio.get_event_loop().run_until_complete(orch.process_all_teams())
    assert 'teamA' in results
    assert isinstance(results['teamA'], list)
    assert len(results['teamA']) == 1
