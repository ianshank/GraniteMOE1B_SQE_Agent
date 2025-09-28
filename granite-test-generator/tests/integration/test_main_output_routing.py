import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_main_generates_per_team_files(tmp_path: Path, monkeypatch):
    """End-to-end: main pipeline writes team-specific JSON outputs.

    This test avoids heavy model paths by replacing the agent with a stub that
    returns minimal test cases. It sets an integration config via environment
    variable to register two local teams, and changes CWD to a temp dir so that
    outputs are contained under that directory.
    """
    # Arrange: create local requirement files for two teams
    team_a = tmp_path / "playback_team"
    team_b = tmp_path / "ads_team"
    team_a.mkdir(parents=True)
    team_b.mkdir(parents=True)
    (team_a / "story.md").write_text("Playback: user can play video\nAs a user...", encoding="utf-8")
    (team_b / "story.md").write_text("Ads: user sees ad\nAs a user...", encoding="utf-8")

    # Integration configuration file
    integ_cfg = tmp_path / "integration_config.yaml"
    integ_cfg.write_text(
        f"""
teams:
  - name: playback_team
    connector:
      type: local
      path: {team_a}
  - name: ads_team
    connector:
      type: local
      path: {team_b}
""".strip(),
        encoding="utf-8",
    )

    # Ensure main merges this integration file and writes output under tmp_path
    monkeypatch.setenv("INTEGRATION_CONFIG_PATH", str(integ_cfg))
    monkeypatch.chdir(tmp_path)

    # Import after env is set
    from src.main import GraniteTestCaseGenerator
    from src.integration.workflow_orchestrator import WorkflowOrchestrator
    from src.models.test_case_schemas import (
        TestCase, TestStep, TestCasePriority, TestCaseType,
    )

    # Use a minimal model config without 'teams' so integration config is merged
    minimal_model_cfg = tmp_path / "model_config.yaml"
    minimal_model_cfg.write_text('model_name: "ibm-granite/granite-3.0-1b-a400m-instruct"\n', encoding="utf-8")

    gen = GraniteTestCaseGenerator(config_path=str(minimal_model_cfg))
    await gen.initialize_system()

    # Stub the agent to return minimal cases without heavy dependencies
    class StubAgent:
        async def generate_test_cases_for_team(self, team, requirements):  # noqa: D401
            return [
                TestCase(
                    id=f"{team}-001",
                    summary="Auto",
                    priority=TestCasePriority.MEDIUM,
                    test_type=TestCaseType.FUNCTIONAL,
                    steps=[TestStep(step_number=1, action="A", expected_result="R")],
                    expected_results="",
                    requirements_traced=[],
                    team_context=team,
                )
            ]

    stub = StubAgent()
    gen.components["orchestrator"] = WorkflowOrchestrator(stub)

    # Register teams from integration config and generate
    gen.register_teams()
    results = await gen.generate_test_cases()

    # Assert results keyed by team
    assert set(results.keys()) == {"playback_team", "ads_team"}

    # Assert files written per team under output/
    out_dir = tmp_path / "output"
    pb_file = out_dir / "playback_team_test_cases.json"
    ads_file = out_dir / "ads_team_test_cases.json"
    assert pb_file.exists() and ads_file.exists()

    pb = json.loads(pb_file.read_text(encoding="utf-8"))
    ads = json.loads(ads_file.read_text(encoding="utf-8"))
    assert isinstance(pb, list) and isinstance(ads, list)
    assert pb and pb[0]["id"].startswith("playback_team-")
    assert ads and ads[0]["id"].startswith("ads_team-")
