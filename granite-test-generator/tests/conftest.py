import sys
import types
from pathlib import Path

# Ensure repo root and package root are on sys.path so `src` imports resolve
_this = Path(__file__).resolve()
pkg_root = _this.parents[1]           # granite-test-generator/tests -> granite-test-generator
repo_root = pkg_root.parent           # project root
for p in (str(pkg_root), str(repo_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Lightweight stubs for optional heavy deps to keep test import-time stable
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
        return "[TEST_CASE][SUMMARY]stub[/SUMMARY][INPUT_DATA]{}[/INPUT_DATA][STEPS]1. s -> e[/STEPS][EXPECTED]ok[/EXPECTED][/TEST_CASE]"
    sys.modules["mlx_lm"].generate = _dummy_generate

# Provide a stub chromadb module so tests can monkeypatch PersistentClient
if "chromadb" not in sys.modules:
    chroma = types.ModuleType("chromadb")
    def _pc(*args, **kwargs):
        raise Exception("chromadb stub called without override")
    chroma.PersistentClient = _pc  # type: ignore
    sys.modules["chromadb"] = chroma
