import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import time
import json

from src.data.rag_retriever import RAGRetriever
from src.utils.kv_cache import KVCache
from src.utils.chunking import DocumentChunk


class TestKVCache:
    """Comprehensive tests for KVCache functionality"""
    
    def test_kvcache_metadata_roundtrip(self, tmp_path: Path):
        """Test metadata persistence across cache instances"""
        cache = KVCache(cache_dir=str(tmp_path))
        key = cache.store("content", {"a": 1}, response="resp", tags=["tag1"])
        entry = cache.retrieve("content", {"a": 1})
        assert entry and entry["response"] == "resp"
        # Metadata persisted
        cache2 = KVCache(cache_dir=str(tmp_path))
        assert key in cache2.metadata
    
    def test_kvcache_eviction(self, tmp_path: Path):
        """Test LRU eviction when cache exceeds max size"""
        cache = KVCache(cache_dir=str(tmp_path), max_size=3)
        
        # Store 4 items, should evict the oldest
        keys = []
        for i in range(4):
            time.sleep(0.01)  # Ensure different timestamps
            key = cache.store(f"content{i}", {"idx": i}, response=f"resp{i}")
            keys.append(key)
        
        # First key should be evicted
        assert cache.retrieve("content0", {"idx": 0}) is None
        # Others should exist
        for i in range(1, 4):
            assert cache.retrieve(f"content{i}", {"idx": i}) is not None
        
        # Metadata should only have 3 entries
        assert len(cache.metadata) == 3
    
    def test_kvcache_tag_retrieval(self, tmp_path: Path):
        """Test retrieving entries by tags"""
        cache = KVCache(cache_dir=str(tmp_path))
        
        # Store entries with different tags
        cache.store("content1", {"a": 1}, response="resp1", tags=["test", "unit"])
        cache.store("content2", {"b": 2}, response="resp2", tags=["test", "integration"])
        cache.store("content3", {"c": 3}, response="resp3", tags=["unit"])
        
        # Retrieve by single tag
        test_entries = cache.retrieve_by_tags(["test"])
        assert len(test_entries) == 2
        
        # Retrieve by multiple tags 
        unit_entries = cache.retrieve_by_tags(["unit"])
        assert len(unit_entries) == 2
        
        # Non-existent tag
        empty_entries = cache.retrieve_by_tags(["nonexistent"])
        assert len(empty_entries) == 0
    
    def test_kvcache_edge_cases(self, tmp_path: Path):
        """Test edge cases: empty content, special characters, large context"""
        cache = KVCache(cache_dir=str(tmp_path))
        
        # Empty content
        key1 = cache.store("", {"empty": True}, response="empty_resp")
        assert cache.retrieve("", {"empty": True})["response"] == "empty_resp"
        
        # Special characters in content
        special_content = "Test with special chars: !@#$%^&*()_+{}[]|\\:\";<>?,./~`"
        key2 = cache.store(special_content, {"special": True}, response="special_resp")
        assert cache.retrieve(special_content, {"special": True})["response"] == "special_resp"
        
        # Large context dictionary
        large_context = {f"key_{i}": f"value_{i}" for i in range(100)}
        key3 = cache.store("large", large_context, response="large_resp")
        assert cache.retrieve("large", large_context)["response"] == "large_resp"
    
    def test_kvcache_corrupted_file_handling(self, tmp_path: Path):
        """Test graceful handling of corrupted cache files"""
        cache = KVCache(cache_dir=str(tmp_path))
        
        # Store valid entry
        key = cache.store("valid", {"a": 1}, response="valid_resp")
        
        # Corrupt the pickle file
        cache_file = tmp_path / f"{key}.pkl"
        with open(cache_file, 'w') as f:
            f.write("corrupted data")
        
        # Should return None for corrupted entry
        assert cache.retrieve("valid", {"a": 1}) is None
        
        # Metadata should still exist
        assert key in cache.metadata


class TestRAGRetriever:
    """Tests for RAG retriever with fallback scenarios"""
    
    def test_rag_fallback_no_embeddings(self, monkeypatch):
        """Test BM25-only fallback when embeddings are unavailable"""
        # Force embeddings unavailable
        monkeypatch.setattr("src.data.rag_retriever.HAS_HF_EMBEDDINGS", False)
        retriever = RAGRetriever()
        assert retriever.embeddings is None
        
        # Prepare dummy chunks
        chunks = [
            DocumentChunk(content="login flow implementation", metadata={}, 
                         chunk_id="c1", source_type="req", team_context="team"),
            DocumentChunk(content="authentication process", metadata={}, 
                         chunk_id="c2", source_type="req", team_context="team")
        ]
        retriever.index_documents(chunks)
        
        # BM25 should still work
        res = retriever.retrieve_relevant_context("login", team_context="team", k=2)
        assert len(res) > 0
        # Check that at least one of the expected documents is retrieved
        retrieved_contents = [r["content"] for r in res]
        assert "login flow implementation" in retrieved_contents or "authentication process" in retrieved_contents
    
    def test_rag_chromadb_persistence_failure(self, monkeypatch, tmp_path: Path):
        """Test fallback when ChromaDB fails to initialize"""
        # Mock ChromaDB to raise exception
        def mock_chromadb_client(*args, **kwargs):
            raise Exception("ChromaDB initialization failed")
        
        monkeypatch.setattr("chromadb.PersistentClient", mock_chromadb_client)
        
        # Should still initialize without ChromaDB
        retriever = RAGRetriever()
        assert retriever.chroma_client is None
        assert retriever.collection is None
    
    @patch('langchain.embeddings.HuggingFaceEmbeddings')
    def test_rag_embedding_initialization_failure(self, mock_hf_embeddings):
        """Test fallback when HuggingFace embeddings fail to load"""
        # Mock embedding initialization to fail
        mock_hf_embeddings.side_effect = Exception("Model download failed")
        
        retriever = RAGRetriever()
        assert retriever.embeddings is None
        
        # Should still work with BM25 only
        chunks = [DocumentChunk(content="test content", metadata={}, 
                               chunk_id="c1", source_type="req", team_context="team")]
        retriever.index_documents(chunks)
        assert retriever.bm25_retriever is not None
    
    def test_rag_empty_document_handling(self):
        """Test handling of empty document lists"""
        retriever = RAGRetriever()
        
        # Index empty list
        retriever.index_documents([])
        
        # Retrieval should return empty list
        res = retriever.retrieve_relevant_context("query")
        assert res == []
    
    def test_rag_duplicate_content_deduplication(self):
        """Test deduplication of identical content in results"""
        retriever = RAGRetriever()
        
        # Index duplicate content
        chunks = [
            DocumentChunk(content="duplicate content", metadata={"id": 1}, 
                         chunk_id="c1", source_type="req", team_context="team"),
            DocumentChunk(content="duplicate content", metadata={"id": 2}, 
                         chunk_id="c2", source_type="req", team_context="team"),
            DocumentChunk(content="unique content", metadata={"id": 3}, 
                         chunk_id="c3", source_type="req", team_context="team")
        ]
        retriever.index_documents(chunks)
        
        # Should deduplicate in results
        res = retriever.retrieve_relevant_context("content", k=5)
        contents = [r["content"] for r in res]
        assert len(set(contents)) == len(contents)  # No duplicates
