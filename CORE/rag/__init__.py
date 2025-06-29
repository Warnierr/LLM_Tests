"""Module RAG (Retrieval-Augmented Generation) pour Nina"""

from .simple_rag import SimpleRAGManager

# Alias pour faciliter l'utilisation
RAGManager = SimpleRAGManager

__all__ = ['RAGManager', 'SimpleRAGManager'] 