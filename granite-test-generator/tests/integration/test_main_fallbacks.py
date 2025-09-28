import asyncio
import json
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_main_fallback_no_teams_no_inputs(tmp_path: Path, monkeypatch):
    # Minimal model config without teams
    model_cfg = tmp_path / "model_config.yaml"
    model_cfg.write_text('model_name: "ibm-granite/granite-3.0-1b-a400m-instruct"\n', encoding="utf-8")

    from src.main import GraniteTestCaseGenerator

    monkeypatch.chdir(tmp_path)
    gen = GraniteTestCaseGenerator(config_path=str(model_cfg))
    await gen.initialize_system()
    await gen.setup_data_pipeline()  # no data dir

    # Do not register teams; trigger fallback logic (should find no local inputs)
    results = await gen.generate_test_cases()
    assert results == {}
    out_dir = tmp_path / "output"
    assert out_dir.exists()
    # Quality report exists and indicates no_data
    report = json.loads((out_dir / "quality_report.json").read_text(encoding="utf-8"))
    assert report["report_status"] == "no_data"


@pytest.mark.asyncio
async def test_main_fallback_from_local_requirements(tmp_path: Path, monkeypatch):
    # Minimal model config without teams
    model_cfg = tmp_path / "model_config.yaml"
    model_cfg.write_text('model_name: "ibm-granite/granite-3.0-1b-a400m-instruct"\n', encoding="utf-8")

    from src.main import GraniteTestCaseGenerator

    # Create local requirements
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "requirements").mkdir(parents=True)
    (tmp_path / "data" / "requirements" / "r1.txt").write_text("Login feature", encoding="utf-8")

    # Isolate the test from any global integration config
    monkeypatch.setenv("INTEGRATION_CONFIG_PATH", "")

    gen = GraniteTestCaseGenerator(config_path=str(model_cfg))
    await gen.initialize_system()
    await gen.setup_data_pipeline()

    # Do not register teams; fallback should generate default test cases
    results = await gen.generate_test_cases()
    assert "default" in results
    assert len(results["default"]) >= 1
    out_file = tmp_path / "output" / "default_test_cases.json"
    assert out_file.exists()

