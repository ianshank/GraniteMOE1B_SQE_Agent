import pytest
import asyncio
import sys
import pathlib
import types

# Ensure project root on path
root_dir = pathlib.Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Stub heavy dependency langchain before importing code under test
if "langchain" not in sys.modules:
    stub_lc = types.ModuleType("langchain")
    sys.modules["langchain"] = stub_lc
    sys.modules["langchain.agents"] = types.ModuleType("langchain.agents")
    sys.modules["langchain.tools"] = types.ModuleType("langchain.tools")
    sys.modules["langchain.schema"] = types.ModuleType("langchain.schema")
    setattr(sys.modules["langchain.agents"], "AgentType", object)
    def _dummy(*args, **kwargs):
        return None
    setattr(sys.modules["langchain.agents"], "initialize_agent", _dummy)
    class _StubTool:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get("name")
        def __call__(self, *args, **kwargs):
            return None
    setattr(sys.modules["langchain.tools"], "Tool", _StubTool)
    setattr(sys.modules["langchain.schema"], "AgentAction", object)
    setattr(sys.modules["langchain.schema"], "AgentFinish", object)

# Provide a base stub for mlx_lm so we can monkeypatch generate during tests
if "mlx_lm" not in sys.modules:
    sys.modules["mlx_lm"] = types.ModuleType("mlx_lm")
    def _placeholder_generate(*args, **kwargs):
        return ""
    sys.modules["mlx_lm"].generate = _placeholder_generate

from src.agents.generation_agent import TestGenerationAgent


class _StubTrainer:
    """Trainer stub to satisfy TestGenerationAgent dependencies."""
    def __init__(self):
        self.mlx_model = None
        self.mlx_tokenizer = None
    def load_model_for_inference(self):
        # No actual model; the agent only forwards objects to mlx_lm.generate
        self.mlx_model = object()
        self.mlx_tokenizer = object()


class _StubRetriever:
    def retrieve_relevant_context(self, query: str, k: int = 3):
        return []


class _StubCache:
    def get_cached_response(self, query: str, team: str):
        return None
    def cache_response(self, **kwargs):
        pass


def _make_agent() -> TestGenerationAgent:
    return TestGenerationAgent(_StubTrainer(), _StubRetriever(), _StubCache())


@pytest.mark.asyncio
@pytest.mark.regression
async def test_generation_fallback_when_missing_tags(monkeypatch):
    """If the model returns text without required tags, agent should fallback."""

    # Force mlx_lm.generate to return invalid output (no tags)
    def _gen(_model, _tok, prompt: str, max_tokens: int = 0):
        assert "[TEST_CASE]" in prompt  # sanity: prompt includes structure
        return "This is not a structured test case."
    monkeypatch.setattr(sys.modules["mlx_lm"], "generate", _gen)

    agent = _make_agent()
    cases = await agent.generate_test_cases_for_team(
        team_name="qa",
        requirements=["# Login feature\nAs a user, I can login."]
    )

    assert len(cases) == 1
    tc = cases[0]
    assert tc.summary.startswith("Test ")
    assert len(tc.steps) >= 1
    assert "should successfully implement" in tc.expected_results


@pytest.mark.regression
def test_has_required_tags_private_helper():
    agent = _make_agent()
    valid = (
        "[TEST_CASE][SUMMARY]x[/SUMMARY][INPUT_DATA]{}[/INPUT_DATA]"
        "[STEPS]1. s -> e\n2. s2 -> e2[/STEPS][EXPECTED]ok[/EXPECTED][/TEST_CASE]"
    )
    invalid = "[SUMMARY]x[/SUMMARY] missing steps and expected"
    assert agent._has_required_tags(valid) is True
    assert agent._has_required_tags(invalid) is False
