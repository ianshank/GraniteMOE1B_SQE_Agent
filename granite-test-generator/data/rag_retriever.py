from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from typing import List, Dict, Any
import chromadb

class RAGRetriever:
    """Retrieval-Augmented Generation system for test case generation"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
        # Initialize ChromaDB for persistent storage
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection("requirements")
    
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
        
        # Create FAISS vectorstore
        texts = [chunk.content for chunk in chunks]
        self.vectorstore = FAISS.from_texts(
            texts, 
            self.embeddings, 
            metadatas=metadatas
        )
        
        # Create BM25 retriever for keyword matching
        self.bm25_retriever = BM25Retriever.from_texts(texts)
        
        # Combine both retrievers
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vectorstore.as_retriever(), self.bm25_retriever],
            weights=[0.7, 0.3]  # Favor semantic similarity slightly
        )
        
        # Store in ChromaDB for persistence
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def retrieve_relevant_context(self, query: str, team_context: str = None, 
                                k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for test case generation"""
        if not self.ensemble_retriever:
            raise ValueError("No documents indexed. Call index_documents first.")
        
        # Enhanced query with team context
        enhanced_query = f"{query} team:{team_context}" if team_context else query
        
        docs = self.ensemble_retriever.get_relevant_documents(enhanced_query)
        
        # Filter by team context if specified
        filtered_docs = []
        for doc in docs[:k]:
            if team_context is None or doc.metadata.get('team_context') == team_context:
                filtered_docs.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'relevance_score': getattr(doc, 'score', 0.0)
                })
        
        return filtered_docs
