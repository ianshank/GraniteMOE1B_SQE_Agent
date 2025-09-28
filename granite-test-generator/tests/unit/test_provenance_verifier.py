import types
import sys
import pathlib


# Ensure project root on path for `src` imports when run via run_tests.py
root_dir = pathlib.Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def _install_minimal_stubs():
    """Install minimal stubs for optional heavy deps to import the agent module.

    This does NOT mock generation; it only provides modules so the import succeeds.
    """
    if "langchain" not in sys.modules:
        stub_lc = types.ModuleType("langchain")
        sys.modules["langchain"] = stub_lc
        sys.modules["langchain.agents"] = types.ModuleType("langchain.agents")
        sys.modules["langchain.tools"] = types.ModuleType("langchain.tools")
        sys.modules["langchain.schema"] = types.ModuleType("langchain.schema")

        # Provide minimal attributes consumed by the agent during __init__
        setattr(sys.modules["langchain.agents"], "AgentType", object)

        class _Tool:
            def __init__(self, *args, **kwargs):
                self.name = kwargs.get("name")

            def __call__(self, *args, **kwargs):
                return None

        setattr(sys.modules["langchain.tools"], "Tool", _Tool)
        setattr(sys.modules["langchain.schema"], "AgentAction", object)
        setattr(sys.modules["langchain.schema"], "AgentFinish", object)

    if "mlx_lm" not in sys.modules:
        # Only needed to satisfy import; test does not invoke generation
        sys.modules["mlx_lm"] = types.ModuleType("mlx_lm")
        sys.modules["mlx_lm"].generate = lambda *a, **k: ""


def test_provenance_overlap_verifier_true_false():
    _install_minimal_stubs()

    from src.agents.generation_agent import TestGenerationAgent

    class _StubTrainer:
        def load_model_for_inference(self):
            pass

    class _StubRetriever:
        def retrieve_relevant_context(self, query: str, k: int = 3):
            return []

    class _StubCache:
        def get_cached_response(self, query: str, team: str):
            return None

        def cache_response(self, **kwargs):
            pass

    agent = TestGenerationAgent(_StubTrainer(), _StubRetriever(), _StubCache(), enforce_strict_provenance=True)

    # Positive overlap case: generated text shares tokens with requirement/context
    req = "User can login to system"
    ctx = "Login page has username and password fields"
    gen = "[TEST_CASE][SUMMARY]login[/SUMMARY][STEPS]1. login -> ok[/STEPS][EXPECTED]ok[/EXPECTED]"
    assert agent._verify_against_context(req, ctx, gen) is True

    # Negative case: no overlap
    req2 = "Process CSV uploads"
    ctx2 = "Parse fields and validate structure"
    gen2 = "[TEST_CASE][SUMMARY]random[/SUMMARY][STEPS]1. unrelated -> none[/STEPS][EXPECTED]x[/EXPECTED]"
    assert agent._verify_against_context(req2, ctx2, gen2) is False
