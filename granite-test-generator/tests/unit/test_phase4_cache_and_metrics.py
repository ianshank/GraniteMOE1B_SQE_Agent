from __future__ import annotations

from typing import Any, Dict, List

from src.agents.generation_agent import TestGenerationAgent


class _StubKV:
    def __init__(self):
        self._s: Dict[str, Dict[str, Any]] = {}

    def _generate_key(self, content: str, context: Dict[str, Any]) -> str:
        return f"{content}|{sorted(context.items())}"

    def store(self, content: str, context: Dict[str, Any], embedding=None, response=None, tags=None) -> str:
        key = self._generate_key(content, context)
        self._s[key] = {
            'content': content,
            'context': context,
            'response': response,
            'tags': tags or [],
        }
        return key

    def retrieve(self, content: str, context: Dict[str, Any]):
        key = self._generate_key(content, context)
        return self._s.get(key)


class _StubCAG:
    def __init__(self, kv):
        self.kv_cache = kv


class _StubModel:
    mlx_model = True
    mlx_tokenizer = True

    def load_model_for_inference(self):
        pass


class _StubRAG:
    def __init__(self):
        self.calls: List[str] = []

    def retrieve_code_snippets(self, query: str, team_context: str = None, language_filter: str = None, k: int = 10, rank_weights: Dict[str, float] | None = None):
        self.calls.append(query)
        return []


def _build_agent():
    kv = _StubKV()
    agent = TestGenerationAgent(
        granite_trainer=_StubModel(),
        rag_retriever=_StubRAG(),
        cag_cache=_StubCAG(kv),
        settings={
            'rag': {
                'code_indexing': {
                    'enabled': True,
                    'max_results': 5,
                    'rank_weights': {'similarity': 0.4, 'quality': 0.3, 'reusability': 0.3},
                }
            }
        },
    )
    return agent


def test_phase4_cache_hit_skips_retrieval():
    agent = _build_agent()
    # First call: miss, will store empty list
    snippets1 = agent._get_code_snippets_with_cache(team_name="teamA", requirement="API test", language_hint="python", k=3, weights=None)
    assert snippets1 == []
    # Second call: hit
    snippets2 = agent._get_code_snippets_with_cache(team_name="teamA", requirement="API test", language_hint="python", k=3, weights=None)
    assert snippets2 == []
    assert agent._metrics['cache_hits'] >= 1


def test_phase4_metrics_populated():
    agent = _build_agent()
    _ = agent._get_code_snippets_with_cache(team_name="teamB", requirement="Login error handling", language_hint="python", k=2, weights=None)
    m = agent._metrics
    assert m['rag_searches'] >= 1
    assert m['last_search_time_ms'] >= 0.0

