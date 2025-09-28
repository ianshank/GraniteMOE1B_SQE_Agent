import pytest
import sys
import pathlib
import types

# Ensure project root on path
root_dir = pathlib.Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Stubs for heavy deps
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

if "mlx_lm" not in sys.modules:
    sys.modules["mlx_lm"] = types.ModuleType("mlx_lm")
    def _dummy_generate(*args, **kwargs):
        return "[TEST_CASE][SUMMARY]ok[/SUMMARY][INPUT_DATA]{}[/INPUT_DATA][STEPS]1. s -> e[/STEPS][EXPECTED]ok[/EXPECTED][/TEST_CASE]"
    sys.modules["mlx_lm"].generate = _dummy_generate

from src.agents.generation_agent import TestGenerationAgent


class _StubTrainer:
    def load_model_for_inference(self):
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


@pytest.mark.asyncio
@pytest.mark.regression
async def test_generate_regression_and_e2e_suites():
    agent = TestGenerationAgent(_StubTrainer(), _StubRetriever(), _StubCache())

    requirements = [
        "# Feature A\nAs a user I can A.",
        "# Feature B\nAs a user I can B.",
    ]

    reg_suite = await agent.generate_test_suite_for_team(
        team_name="qa", requirements=requirements, suite_name="Regression Suite", description="Regression"
    )
    e2e_suite = await agent.generate_test_suite_for_team(
        team_name="qa", requirements=requirements, suite_name="E2E Suite", description="End-to-End"
    )

    assert reg_suite.name == "Regression Suite"
    assert e2e_suite.name == "E2E Suite"
    assert len(reg_suite.test_cases) == len(requirements)
    assert len(e2e_suite.test_cases) == len(requirements)
