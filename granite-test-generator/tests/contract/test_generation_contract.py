import sys
import pathlib
import types

import pytest

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

# Stub optional mlx_lm dependency for import-time safety
if "mlx_lm" not in sys.modules:
    sys.modules["mlx_lm"] = types.ModuleType("mlx_lm")
    def _dummy_generate(*args, **kwargs):
        return ""
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


def _make_agent() -> TestGenerationAgent:
    return TestGenerationAgent(_StubTrainer(), _StubRetriever(), _StubCache())


def test_validation_contract_passes_on_structured_output():
    agent = _make_agent()
    text = (
        "[TEST_CASE][SUMMARY]Title[/SUMMARY]"
        "[INPUT_DATA]{}[/INPUT_DATA]"
        "[STEPS]1. action -> expected\n2. next -> ok[/STEPS]"
        "[EXPECTED]ok[/EXPECTED][/TEST_CASE]"
    )
    assert agent._validate_test_case(text).startswith("Validation passed")


def test_validation_contract_fails_on_missing_tags():
    agent = _make_agent()
    text = "[SUMMARY]x[/SUMMARY]"
    out = agent._validate_test_case(text)
    assert out.startswith("Validation failed")
    assert "[STEPS]" in out  # should list missing tags
