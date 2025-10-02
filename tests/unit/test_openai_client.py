import sys
import types
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "granite-test-generator" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from integration import openai_client


class _FakeResponses:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(output_text="synthetic result")


def _build_client(monkeypatch, *, context_window=256):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai_client, "_OPENAI_AVAILABLE", True, raising=False)

    fake_responses = _FakeResponses()

    class _FakeOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.responses = fake_responses

    monkeypatch.setattr(openai_client, "OpenAI", _FakeOpenAI, raising=False)

    client = openai_client.OpenAIClient(api_key="test-key", default_model="gpt-4o-mini")
    client.max_context_tokens = context_window
    return client, fake_responses


def test_generate_response_truncates_prompt_when_exceeding_budget(monkeypatch):
    client, responses = _build_client(monkeypatch, context_window=128)

    long_prompt = "A" * 500
    result = client.generate_response(long_prompt, max_output_tokens=40, model="gpt-4o-mini")

    assert result == "synthetic result"
    assert responses.calls, "OpenAI responses.create should be invoked"

    request = responses.calls[0]
    assert request["max_output_tokens"] <= 40
    assert len(request["input"]) + request["max_output_tokens"] <= client.max_context_tokens
    assert "[TRUNCATED" in request["input"]


def test_generate_response_retains_prompt_within_budget(monkeypatch):
    client, responses = _build_client(monkeypatch, context_window=512)

    prompt = "short prompt"
    result = client.generate_response(prompt, max_output_tokens=60)

    assert result == "synthetic result"
    request = responses.calls[0]
    assert request["input"] == prompt
    assert request["max_output_tokens"] == 60


def test_count_tokens_works_without_tiktoken(monkeypatch):
    client, _ = _build_client(monkeypatch, context_window=128)
    monkeypatch.setattr(openai_client, "_TIKTOKEN_AVAILABLE", False, raising=False)
    monkeypatch.setattr(openai_client, "tiktoken", None, raising=False)

    text = "token"
    assert client.count_tokens(text) == len(text)
