from typing import List

from src.data.rag_retriever import SimpleEnsembleRetriever


class NoContentDoc:
    """Doc object without page_content; __str__ returns a constant.

    This simulates third-party docs that do not expose text content and have
    non-unique string representations, which could break identity logic if
    str(obj) were used as the key.
    """

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:  # Non-unique representation
        return "<doc>"


class DummyRetriever:
    def __init__(self, docs: List[NoContentDoc]):
        self._docs = docs

    def get_relevant_documents(self, _query: str):
        return list(self._docs)


def test_rrf_uses_object_id_when_no_page_content():
    """RRF should fall back to object identity, not str(obj).

    When two docs lack page_content and share identical __str__ output, the
    ensemble must still treat them as distinct entries.
    """
    d1 = NoContentDoc("a")
    d2 = NoContentDoc("b")

    r1 = DummyRetriever([d1])
    r2 = DummyRetriever([d2])

    ens = SimpleEnsembleRetriever([r1, r2])
    out = ens.get_relevant_documents("q")

    # Both docs appear; if str(obj) were used as identity, one would be lost
    # due to deduplication under the same key.
    assert len(out) == 2
    assert d1 in out and d2 in out

