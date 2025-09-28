import pytest


def test_local_only_mode_flag(monkeypatch):
    from src.main import GraniteTestCaseGenerator

    monkeypatch.setenv("GRANITE_LOCAL_ONLY", "true")
    generator = GraniteTestCaseGenerator(config_dict={})

    assert generator.local_only_mode is True

