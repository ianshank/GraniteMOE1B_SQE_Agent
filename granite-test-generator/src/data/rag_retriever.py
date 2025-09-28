import logging
from typing import List, Dict, Any, Optional
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from src.utils.chunking import DocumentChunk

# Conditional import for HuggingFaceEmbeddings
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    HAS_HF_EMBEDDINGS = True
except ImportError:
    logging.warning("Could not import HuggingFaceEmbeddings. RAG will use BM25 only. Install 'sentence-transformers' for full RAG capabilities.")
    HAS_HF_EMBEDDINGS = False

# Conditional import for FAISS  
try:
    from langchain.vectorstores import FAISS
    HAS_FAISS = True
except ImportError:
    logging.warning("Could not import FAISS. RAG will use BM25 only. Install 'faiss-cpu' for full RAG capabilities.")
    HAS_FAISS = False

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
        
        # Initialize ChromaDB for persistent storage
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.chroma_client.get_or_create_collection("requirements")
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
        
        # Create BM25 retriever for keyword matching
        try:
            self.bm25_retriever = BM25Retriever.from_texts(documents, metadatas=metadatas)
            self.bm25_retriever.k = 5  # Set default k
            logger.info(f"BM25 retriever created with {len(documents)} documents.")
        except Exception as e:
            logger.error(f"Failed to create BM25 retriever: {e}.")
            self.bm25_retriever = None
        
        # Combine both retrievers if both are available
        retrievers = []
        if self.vectorstore:
            retrievers.append(self.vectorstore.as_retriever())
        if self.bm25_retriever:
            retrievers.append(self.bm25_retriever)
        
        if retrievers:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=retrievers,
                weights=[0.7, 0.3] if len(retrievers) == 2 else [1.0]  # Adjust weights if only one retriever
            )
            logger.info("Ensemble retriever created.")
        else:
            logger.error("No retrievers could be initialized. Retrieval will not be possible.")
            self.ensemble_retriever = None
        
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
    
    def retrieve_relevant_context(self, query: str, team_context: Optional[str] = None,
                                k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for test case generation"""
        # Enhanced query with team context
        enhanced_query = f"{query} team:{team_context}" if team_context else query
        
        docs = []
        if self.ensemble_retriever:
            docs = self.ensemble_retriever.get_relevant_documents(enhanced_query)
        elif self.bm25_retriever:
            # Direct BM25 retrieval when no ensemble
            docs = self.bm25_retriever.get_relevant_documents(enhanced_query)
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