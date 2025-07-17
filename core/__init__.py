# core/__init__.py
"""
Core module for the UAM Literature Review RAG System.
Contains the main pipeline components organized into separate modules.
"""

from .rag_system import UAMRAGSystem
from .embeddings import EmbeddingManager
from .retrieval import RetrievalEngine
from .reranking import ReRankingEngine
from .generation import ResponseGenerator
from .ingestion import DocumentIngester
from .llm_client import OpenRouterLLM

__version__ = "2.0.0"
__author__ = "UAM Research Team"
__description__ = "Modular RAG system for UAM literature review"

__all__ = [
    'UAMRAGSystem',
    'EmbeddingManager',
    'RetrievalEngine', 
    'ReRankingEngine',
    'ResponseGenerator',
    'DocumentIngester',
    'OpenRouterLLM'
]

# Version information
VERSION_INFO = {
    'major': 2,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

def get_version():
    """Get the current version string"""
    return f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"

def get_info():
    """Get system information"""
    return {
        'name': 'UAM Literature Review RAG System',
        'version': get_version(),
        'description': __description__,
        'modules': __all__
    }