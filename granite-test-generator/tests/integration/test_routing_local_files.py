import asyncio
from types import SimpleNamespace
from pathlib import Path

from src.integration.workflow_orchestrator import WorkflowOrchestrator, TeamConfiguration
from src.integration.team_connectors import LocalFileSystemConnector


def test_orchestrator_with_local_connector_routes_by_team(tmp_path: Path):
    """Requirements from LocalFileSystemConnector route to team-specific results."""
    # Create local requirement
    (tmp_path / "story.md").write_text("Login works\nAs a user...", encoding="utf-8")

    class StubAgent:
        async def generate_test_cases_for_team(self, team, reqs):
            # Create a minimal test case-like object
            return [SimpleNamespace(
                id=f"{team}-001",
                summary="Generated",
                steps=[],
                expected_results="",
                priority=SimpleNamespace(value="medium"),
                test_type=SimpleNamespace(value="functional"),
                requirements_traced=[]
            )]

    orch = WorkflowOrchestrator(StubAgent())
    team_name = "playback_team"
    connector = LocalFileSystemConnector(directory=str(tmp_path), team_name=team_name)
    orch.register_team(TeamConfiguration(team_name=team_name, connector=connector))

    results = asyncio.get_event_loop().run_until_complete(orch.process_all_teams())
    assert team_name in results
    assert isinstance(results[team_name], list) and len(results[team_name]) == 1

