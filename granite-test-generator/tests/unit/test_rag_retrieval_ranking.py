# tests/unit/test_rag_retrieval_ranking.py
from __future__ import annotations

from typing import Any, Dict, List

from src.data.rag_retriever import RAGRetriever


class _FakeRetriever(RAGRetriever):
    def __init__(self):
        # Bypass parent init to avoid external deps
        self.embeddings = None
        self.vectorstore = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self._raw_documents = []
        self._raw_metadatas = []

    def retrieve_relevant_context(self, query: str, team_context: str | None = None, k: int = 5):
        # Return deterministic docs
        docs: List[Dict[str, Any]] = [
            {
                "content": "def a(): pass",
                "metadata": {"language": "python", "quality_score": 0.9, "reusability_score": 0.6, "description": "A", "pattern_type": "function_definition"},
                "relevance_score": 0.2,
            },
            {
                "content": "def b(): pass",
                "metadata": {"language": "python", "quality_score": 0.4, "reusability_score": 0.9, "description": "B", "pattern_type": "function_definition"},
                "relevance_score": 0.8,
            },
        ]
        return docs[:k]


def test_rank_documents_with_weights_changes_order():
    r = _FakeRetriever()
    docs = r.retrieve_relevant_context("x", k=2)

    ranked_sim_heavy = r.rank_documents_with_weights(docs, w_similarity=0.8, w_quality=0.1, w_reuse=0.1)
    ranked_qual_heavy = r.rank_documents_with_weights(docs, w_similarity=0.1, w_quality=0.8, w_reuse=0.1)

    # With similarity heavy, second doc (higher relevance_score=0.8) should win
    assert ranked_sim_heavy[0]["content"].startswith("def b")

    # With quality heavy, first doc (quality_score=0.9) should win
    assert ranked_qual_heavy[0]["content"].startswith("def a")


def test_retrieve_code_snippets_maps_metadata_and_filters_language():
    r = _FakeRetriever()
    snippets = r.retrieve_code_snippets("x", language_filter="python", k=2)
    assert len(snippets) == 2
    assert snippets[0].language.value == "python"
    assert "blended_score" in snippets[0].metadata
    assert "similarity_score" in snippets[0].metadata
