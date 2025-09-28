import logging
from typing import Dict

import pytest


def test_local_only_mode_flag(monkeypatch):
    from src.main import GraniteTestCaseGenerator

    monkeypatch.setenv("GRANITE_LOCAL_ONLY", "true")
    generator = GraniteTestCaseGenerator(config_dict={})

    assert generator.local_only_mode is True


def test_local_only_mode_false(monkeypatch):
    from src.main import GraniteTestCaseGenerator

    monkeypatch.setenv("GRANITE_LOCAL_ONLY", "0")
    generator = GraniteTestCaseGenerator(config_dict={})

    assert generator.local_only_mode is False


def test_local_only_mode_invalid(monkeypatch):
    from src.main import GraniteTestCaseGenerator

    monkeypatch.setenv("GRANITE_LOCAL_ONLY", "maybe")
    with pytest.raises(ValueError, match="GRANITE_LOCAL_ONLY"):
        GraniteTestCaseGenerator(config_dict={})


@pytest.mark.asyncio
async def test_generate_test_cases_uses_debug_logging(caplog, tmp_path):
    from src.main import GraniteTestCaseGenerator

    class _DummyOrchestrator:
        def __init__(self) -> None:
            self.team_configs: Dict[str, object] = {"team-alpha": object()}

        async def process_all_teams(self) -> Dict[str, list]:
            return {"team-alpha": []}

        def generate_quality_report(self) -> Dict[str, object]:
            return {
                "total_test_cases": 0,
                "teams_processed": 1,
                "teams_with_results": 1,
            }

    generator = GraniteTestCaseGenerator(
        config_dict={
            'paths': {'output_dir': str(tmp_path)},
        }
    )
    generator.components['orchestrator'] = _DummyOrchestrator()
    generator.components['has_requirements'] = False

    with caplog.at_level(logging.DEBUG):
        results = await generator.generate_test_cases()

    assert results == {"team-alpha": []}

    debug_messages = [record.message for record in caplog.records if record.levelno == logging.DEBUG]
    assert any("process_all_teams returned" in message for message in debug_messages)
    assert all("DIAGNOSTIC" not in record.message for record in caplog.records)

    # Ensure artifacts are written without error
    output_file = tmp_path / "team-alpha_test_cases.json"
    quality_report_file = tmp_path / "quality_report.json"
    assert output_file.exists()
    assert quality_report_file.exists()
