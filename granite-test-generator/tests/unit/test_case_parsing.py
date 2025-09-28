import pytest
import asyncio
# noqa: E402
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

# Stub optional mlx_lm (Apple MLX) dependency
if "mlx_lm" not in sys.modules:
    sys.modules["mlx_lm"] = types.ModuleType("mlx_lm")
    def _dummy_generate(*args, **kwargs):
        return "[TEST_CASE][SUMMARY]dummy[/SUMMARY][INPUT_DATA]{}[/INPUT_DATA][STEPS]1. step -> ok[/STEPS][EXPECTED]ok[/EXPECTED][/TEST_CASE]"
    sys.modules["mlx_lm"].generate = _dummy_generate

from src.agents.generation_agent import TestGenerationAgent


# ---------------- Stubs -----------------


class _StubTrainer:
    """Trainer stub that forces template fallback."""

    def load_model_for_inference(self):
        self.mlx_model = None
        self.mlx_tokenizer = None


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


# ---------------- Tests -----------------


@pytest.mark.asyncio
@pytest.mark.regression
async def test_template_generation_parsing_unique_ids():
    """Ensure parser returns unique IDs and non-empty steps for different requirements."""

    agent = _make_agent()
    requirements = [
        "User should be able to login with valid credentials",
        "System must ingest uploaded CSV files",
    ]

    cases = await agent.generate_test_cases_for_team("qa", requirements)

    assert len(cases) == 2

    # IDs must differ
    ids = {tc.id for tc in cases}
    assert len(ids) == 2, "IDs should be unique per test case"

    for tc in cases:
        assert tc.summary, "Summary should not be empty"
        assert len(tc.steps) >= 1, "Steps should not be empty"
        assert tc.expected_results, "Expected results should not be empty"
