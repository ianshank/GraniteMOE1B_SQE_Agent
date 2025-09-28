import logging
from typing import List, Dict, Any, Optional
from src.utils.chunking import DocumentChunk

# chromadb is optional in CI/test environments. Provide a safe shim so imports
# and monkeypatches in tests continue to work even when the package is absent.
try:  # pragma: no cover - exercised indirectly in tests
    import chromadb  # type: ignore
    HAS_CHROMADB = True
except Exception:  # pragma: no cover
    import sys
    import types

    HAS_CHROMADB = False
    chromadb = types.ModuleType("chromadb")  # type: ignore

    class _MissingPersistentClient:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("chromadb not available")

    chromadb.PersistentClient = _MissingPersistentClient  # type: ignore[attr-defined]
    sys.modules["chromadb"] = chromadb  # Ensure monkeypatch can locate it

# LangChain v0.3+: modules reorganized. Import with fallbacks for compatibility.
try:  # BM25 lives in community package
    from langchain_community.retrievers import BM25Retriever  # type: ignore
except Exception:  # pragma: no cover
    BM25Retriever = None  # type: ignore

try:  # FAISS lives in community package
    from langchain_community.vectorstores import FAISS  # type: ignore
    HAS_FAISS = True
except Exception:  # pragma: no cover
    FAISS = None  # type: ignore
    HAS_FAISS = False

try:  # Ensemble retriever remains in core retrievers
    from langchain.retrievers import EnsembleRetriever  # type: ignore
except Exception:  # pragma: no cover
    EnsembleRetriever = None  # type: ignore


class SimpleEnsembleRetriever:
    """Lightweight fallback to combine multiple retrievers without LC dependency.

    Uses a simple reciprocal-rank fusion over results. If only one retriever is
    provided, it delegates directly to that retriever.
    """

    def __init__(self, retrievers: list, weights: Optional[list] = None, c: int = 60):
        self.retrievers = retrievers
        self.weights = weights or [1.0 / max(1, len(retrievers))] * max(1, len(retrievers))
        self.c = c

    def get_relevant_documents(self, query: str):
        if not self.retrievers:
            return []
        if len(self.retrievers) == 1:
            return self.retrievers[0].get_relevant_documents(query)

        # Reciprocal Rank Fusion (RRF)
        from collections import defaultdict
        docs_per_ret = [ret.get_relevant_documents(query) for ret in self.retrievers]
        scores = defaultdict(float)
        all_docs = []
        seen = set()
        for r_idx, docs in enumerate(docs_per_ret):
            w = self.weights[r_idx] if r_idx < len(self.weights) else 1.0
            for rank, doc in enumerate(docs, start=1):
                # Use page_content as identity, else fall back to object id.
                # Avoid str(obj) because some implementations may return non-unique
                # representations, which would corrupt RRF fusion.
                key = getattr(doc, "page_content", None)
                if key is None:
                    key = f"obj:{id(doc)}"
                scores[key] += w / (self.c + rank)
                if key not in seen:
                    all_docs.append(doc)
                    seen.add(key)
        # Sort by fused score
        all_docs.sort(key=lambda d: scores.get(getattr(d, "page_content", str(d)), 0.0), reverse=True)
        return all_docs


class _FallbackDocument:
    """Minimal document object compatible with LangChain docs in tests.

    Attributes mirror the subset used in this project: `page_content`,
    `metadata`, and optional `score`.
    """

    def __init__(self, page_content: str, metadata: Dict[str, Any], score: float = 0.0):
        self.page_content = page_content
        self.metadata = metadata
        self.score = score


class SimpleKeywordRetriever:
    """Lightweight keyword-based retriever used when BM25 is unavailable.

    This avoids a hard dependency on `langchain_community` for unit tests. It
    performs a trivial keyword match and ranks by occurrence count.
    """

    def __init__(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, k: int = 5):
        self.texts = texts
        self.metadatas = metadatas or [{} for _ in texts]
        self.k = k

    @classmethod
    def from_texts(cls, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        return cls(texts=texts, metadatas=metadatas)

    def get_relevant_documents(self, query: str) -> List[_FallbackDocument]:
        tokens = [t for t in query.lower().split() if t]
        scored: List[_FallbackDocument] = []
        for idx, content in enumerate(self.texts):
            lc = content.lower()
            score = sum(lc.count(tok) for tok in tokens) if tokens else 0.0
            # If no tokens, provide a minimal non-zero score to keep first k
            score = float(score)
            scored.append(_FallbackDocument(page_content=content, metadata=self.metadatas[idx], score=score))
        # Sort by score desc, then stable order
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[: self.k]

# Conditional import for HuggingFaceEmbeddings (LangChain package locations)
try:  # Newer recommended import path
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    HAS_HF_EMBEDDINGS = True
except Exception:  # pragma: no cover
    try:  # External split package some environments use
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
        HAS_HF_EMBEDDINGS = True
    except Exception:
        try:
            # Deprecated fallback path; avoid when possible to prevent warnings
            from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
            HAS_HF_EMBEDDINGS = True
        except Exception:
            logging.warning(
                "Could not import HuggingFaceEmbeddings. RAG will use BM25 only. "
                "Install 'sentence-transformers' and 'langchain-community' (or 'langchain-huggingface') for full RAG capabilities."
            )
            HAS_HF_EMBEDDINGS = False
            # Ensure symbol exists so tests can patch it
            class HuggingFaceEmbeddings:  # type: ignore
                pass

logger = logging.getLogger(__name__)

class RAGRetriever:
    """Retrieval-Augmented Generation system for test case generation"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = None
        if HAS_HF_EMBEDDINGS:
            try:
                self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
                logger.info(f"HuggingFaceEmbeddings initialized with model: {embedding_model}")
            except Exception as e:
                logger.error(f"Failed to initialize HuggingFaceEmbeddings: {e}. Falling back to BM25 only.")
                self.embeddings = None
        else:
            logger.info("HuggingFaceEmbeddings not available. RAG will use BM25 only.")
        
        self.vectorstore = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        # Keep a copy of raw texts and metadatas for robust fallbacks
        self._raw_documents: List[str] = []
        self._raw_metadatas: List[Dict[str, Any]] = []
        
        # Initialize ChromaDB for persistent storage (optional)
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")  # type: ignore[attr-defined]
            self.collection = self.chroma_client.get_or_create_collection("requirements")  # type: ignore[assignment]
            logger.info("ChromaDB initialized for persistent storage.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}. Persistence will not be used.")
            self.chroma_client = None
            self.collection = None
    
    def index_documents(self, chunks: List[DocumentChunk]):
        """Index document chunks for retrieval"""
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            documents.append(chunk.content)
            metadatas.append({
                'source_type': chunk.source_type,
                'team_context': chunk.team_context,
                **chunk.metadata
            })
            ids.append(chunk.chunk_id)
        
        if not documents:
            logger.warning("No documents to index.")
            return
        
        # Create FAISS vectorstore if embeddings are available
        if self.embeddings and HAS_FAISS:
            try:
                self.vectorstore = FAISS.from_texts(
                    documents,
                    self.embeddings,
                    metadatas=metadatas
                )
                logger.info(f"FAISS vectorstore created with {len(documents)} documents.")
            except Exception as e:
                logger.error(f"Failed to create FAISS vectorstore: {e}. Falling back to BM25 only.")
                self.vectorstore = None
        else:
            logger.info("FAISS not available or embeddings not initialized. Vectorstore will not be used.")
        
        # Create BM25 (or fallback) retriever for keyword matching
        if BM25Retriever is not None:  # type: ignore[truthy-bool]
            try:
                self.bm25_retriever = BM25Retriever.from_texts(documents, metadatas=metadatas)  # type: ignore[attr-defined]
                self.bm25_retriever.k = 5  # type: ignore[attr-defined]
                logger.info(f"BM25 retriever created with {len(documents)} documents.")
            except Exception as e:
                logger.error(f"Failed to create BM25 retriever: {e}. Falling back to SimpleKeywordRetriever.")
                self.bm25_retriever = SimpleKeywordRetriever.from_texts(documents, metadatas=metadatas)
        else:
            logger.info("BM25Retriever not available. Using SimpleKeywordRetriever fallback.")
            self.bm25_retriever = SimpleKeywordRetriever.from_texts(documents, metadatas=metadatas)
        
        # Combine both retrievers if available; otherwise use the single one
        retrievers = []
        if self.vectorstore:
            retrievers.append(self.vectorstore.as_retriever())
        if self.bm25_retriever:
            retrievers.append(self.bm25_retriever)

        if not retrievers:
            logger.error("No retrievers could be initialized. Retrieval will not be possible.")
            self.ensemble_retriever = None
        elif len(retrievers) == 1:
            # Use the single retriever directly
            self.ensemble_retriever = retrievers[0]
            logger.info("Initialized single retriever (no ensemble).")
        else:
            weights = [0.7, 0.3]
            if EnsembleRetriever is not None:
                self.ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)  # type: ignore
                logger.info("Ensemble retriever created (LangChain).")
            else:
                self.ensemble_retriever = SimpleEnsembleRetriever(retrievers=retrievers, weights=weights)
                logger.info("Ensemble retriever created (fallback).")
        
        # Store in ChromaDB for persistence
        if self.collection:
            try:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Documents added to ChromaDB collection: {self.collection.name}")
            except Exception as e:
                logger.error(f"Failed to add documents to ChromaDB: {e}.")
        
        # Save raw docs for later fallback retrieval
        self._raw_documents = documents
        self._raw_metadatas = metadatas
    
    def retrieve_relevant_context(self, query: str, team_context: Optional[str] = None,
                                k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for test case generation"""
        # Enhanced query with team context
        enhanced_query = f"{query} team:{team_context}" if team_context else query
        
        docs = []
        if self.ensemble_retriever:
            # LangChain retrievers expose invoke/get_relevant_documents; handle both, with robust fallback
            get_docs = getattr(self.ensemble_retriever, "get_relevant_documents", None)
            try:
                if callable(get_docs):
                    docs = get_docs(enhanced_query)
                else:
                    docs = self.ensemble_retriever.invoke(enhanced_query)  # type: ignore[attr-defined]
            except Exception as e:  # pragma: no cover - environment-specific LC issues
                logger.error(f"Primary retriever failed (`{type(self.ensemble_retriever).__name__}`): {e}. Falling back to keyword scoring.")
                docs = self._keyword_fallback(enhanced_query)
        elif self.bm25_retriever:
            # Direct retrieval when no ensemble
            try:
                docs = self.bm25_retriever.get_relevant_documents(enhanced_query)
            except Exception as e:  # pragma: no cover
                logger.error(f"BM25 retrieval failed: {e}. Falling back to keyword scoring.")
                docs = self._keyword_fallback(enhanced_query)
        else:
            logger.warning("No documents indexed or retrievers initialized. Returning empty context.")
            return []
        
        # Filter by team context if specified and deduplicate
        filtered_docs = []
        seen_content = set()
        for doc in docs:
            if (team_context is None or doc.metadata.get('team_context') == team_context) and doc.page_content not in seen_content:
                filtered_docs.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'relevance_score': getattr(doc, 'score', 0.0)
                })
                seen_content.add(doc.page_content)
            if len(filtered_docs) >= k:  # Cap at k documents
                break
        
        logger.debug(f"Retrieved {len(filtered_docs)} documents for query: '{query}' (team: {team_context})")
        return filtered_docs

    # --------------------- Internal keyword fallback ---------------------
    def _keyword_fallback(self, enhanced_query: str):
        """Compute naive keyword-match scores over raw docs, independent of retriever classes.

        This avoids monkeypatching collisions in tests that replace retriever methods
        at the class level (e.g., BM25 or SimpleKeywordRetriever).
        """
        tokens = [t for t in (enhanced_query or '').lower().split() if t]
        results: List[_FallbackDocument] = []
        for content, meta in zip(getattr(self, '_raw_documents', []), getattr(self, '_raw_metadatas', [])):
            lc = (content or '').lower()
            score = float(sum(lc.count(tok) for tok in tokens))
            results.append(_FallbackDocument(page_content=content, metadata=meta, score=score))
        # Sort and return top-k similar to other retrievers
        results.sort(key=lambda d: d.score, reverse=True)
        return results
