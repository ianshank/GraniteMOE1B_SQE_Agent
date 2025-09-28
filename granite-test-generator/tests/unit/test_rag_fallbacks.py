import pytest
from src.data.rag_retriever import RAGRetriever
from src.utils.chunking import DocumentChunk


def test_retrieval_falls_back_when_primary_errors(monkeypatch):
    """If the primary retriever errors at call time, fallback returns results.

    This simulates environment-specific LangChain issues and ensures our
    robust fallback path still provides relevant documents.
    """
    retriever = RAGRetriever()

    chunks = [
        DocumentChunk(content="user login works", metadata={}, chunk_id="c1", source_type="req", team_context="t"),
        DocumentChunk(content="logout flow", metadata={}, chunk_id="c2", source_type="req", team_context="t"),
    ]
    retriever.index_documents(chunks)

    # Force the active retriever (ensemble or bm25) to raise on query
    target = getattr(retriever, "ensemble_retriever", None) or retriever.bm25_retriever

    def boom(*args, **kwargs):  # type: ignore[unused-argument]
        raise RuntimeError("unexpected retriever failure")

    if hasattr(target, "get_relevant_documents"):
        # Some retrievers (e.g., LangChain BM25) are Pydantic models and
        # prohibit setting new attributes on instances. Patch at the class level.
        if hasattr(target.__class__, "get_relevant_documents"):
            def boom_method(self, *args, **kwargs):  # type: ignore[unused-argument]
                raise RuntimeError("unexpected retriever failure")
            monkeypatch.setattr(target.__class__, "get_relevant_documents", boom_method, raising=True)
        else:
            monkeypatch.setattr(target, "get_relevant_documents", boom)

    results = retriever.retrieve_relevant_context("login", team_context="t", k=2)
    assert isinstance(results, list)
    assert len(results) >= 1
    contents = [r["content"] for r in results]
    assert any("login" in c for c in contents)
