# core/embeddings.py
"""
Embedding Management Module
Handles document and query embeddings for the RAG system.
"""

import logging
from typing import List, Union
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embeddings for documents and queries"""
    
    def __init__(self):
        """Initialize embedding model"""
        logger.info(f"Initializing embedding model: {Config.EMBEDDING_MODEL}")
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        logger.info("Embedding model initialized successfully")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self.embedding_model.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query"""
        return self.embedding_model.embed_query(text)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        # Test with a dummy text
        test_embedding = self.embed_query("test")
        return len(test_embedding)