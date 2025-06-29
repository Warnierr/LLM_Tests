"""
Gestionnaire RAG (Retrieval-Augmented Generation) pour Nina
Intègre ChromaDB et LlamaIndex pour la recherche sémantique avancée
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid

try:
    from llama_index import VectorStoreIndex, ServiceContext, Document
    from llama_index.vector_stores import ChromaVectorStore
    from llama_index.storage.storage_context import StorageContext
    from llama_index.embeddings import HuggingFaceEmbedding
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from sentence_transformers import SentenceTransformer
import numpy as np
from ..utils.telemetry import get_telemetry


class RAGManager:
    """Gestionnaire RAG unifié pour Nina avec fallback gracieux"""
    
    def __init__(
        self,
        persist_directory: str = "MEMORY/rag_store",
        collection_name: str = "nina_documents",
        embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        self.telemetry = get_telemetry()
        self.logger = self.telemetry.get_logger("RAGManager")
        
        # Initialisation des composants
        self._init_components()
        
    def _init_components(self):
        """Initialise les composants RAG avec fallback"""
        self.logger.info("Initialisation du gestionnaire RAG")
        
        # Modèle d'embedding local (toujours disponible)
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Tentative d'initialisation LlamaIndex + Chroma
        self.llama_index = None
        self.chroma_client = None
        self.collection = None
        
        if LLAMA_INDEX_AVAILABLE and CHROMA_AVAILABLE:
            try:
                self._init_llama_index()
                self.logger.info("LlamaIndex + ChromaDB initialisé avec succès")
            except Exception as e:
                self.logger.warning(f"Échec LlamaIndex: {e}, fallback vers ChromaDB seul")
                self._init_chroma_only()
        elif CHROMA_AVAILABLE:
            self._init_chroma_only()
        else:
            self.logger.warning("Ni LlamaIndex ni ChromaDB disponibles, mode dégradé")
            
    def _init_llama_index(self):
        """Initialise LlamaIndex avec ChromaDB comme vector store"""
        # Configuration ChromaDB
        chroma_client = chromadb.PersistentClient(path=str(self.persist_directory))
        chroma_collection = chroma_client.get_or_create_collection(self.collection_name)
        
        # Configuration LlamaIndex
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Embedding model pour LlamaIndex
        embed_model = HuggingFaceEmbedding(model_name=self.embedding_model_name)
        service_context = ServiceContext.from_defaults(embed_model=embed_model)
        
        # Index
        self.llama_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            service_context=service_context
        )
        
    def _init_chroma_only(self):
        """Initialise ChromaDB seul"""
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Ajoute un document au système RAG"""
        doc_id = f"doc_{uuid.uuid4().hex}"
        
        with self.telemetry.trace_operation("rag_add_document", {"doc_id": doc_id}):
            try:
                if self.llama_index:
                    # Mode LlamaIndex
                    document = Document(text=text, metadata=metadata or {}, doc_id=doc_id)
                    self.llama_index.insert(document)
                    
                elif self.collection:
                    # Mode ChromaDB seul
                    embedding = self.embedding_model.encode(text, normalize_embeddings=True)
                    self.collection.add(
                        embeddings=[embedding.tolist()],
                        documents=[text],
                        metadatas=[metadata or {}],
                        ids=[doc_id]
                    )
                else:
                    raise RuntimeError("Aucun backend RAG disponible")
                    
                self.telemetry.record_memory_operation("rag_add", 1)
                self.logger.info("Document ajouté au RAG", doc_id=doc_id)
                return doc_id
                
            except Exception as e:
                self.logger.error("Erreur ajout document RAG", error=str(e))
                raise
                
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Recherche sémantique dans les documents"""
        
        with self.telemetry.trace_operation("rag_search", {"query_length": len(query), "top_k": top_k}):
            try:
                if self.llama_index:
                    # Mode LlamaIndex
                    retriever = self.llama_index.as_retriever(similarity_top_k=top_k)
                    nodes = retriever.retrieve(query)
                    
                    results = []
                    for node in nodes:
                        results.append({
                            "text": node.node.text,
                            "score": node.score,
                            "metadata": node.node.metadata,
                            "doc_id": node.node.doc_id
                        })
                    
                elif self.collection:
                    # Mode ChromaDB seul
                    query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
                    
                    search_results = self.collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=top_k
                    )
                    
                    results = []
                    if search_results and search_results.get('documents'):
                        for i, (doc, distance, metadata, doc_id) in enumerate(zip(
                            search_results['documents'][0],
                            search_results['distances'][0], 
                            search_results['metadatas'][0],
                            search_results['ids'][0]
                        )):
                            score = 1 - distance  # Convert distance to similarity
                            results.append({
                                "text": doc,
                                "score": score,
                                "metadata": metadata,
                                "doc_id": doc_id
                            })
                else:
                    self.logger.warning("Aucun backend RAG disponible pour la recherche")
                    return []
                    
                self.telemetry.record_memory_operation("rag_search", len(results))
                self.logger.info("Recherche RAG effectuée", 
                               query_length=len(query), 
                               results_count=len(results))
                
                return results
                
            except Exception as e:
                self.logger.error("Erreur recherche RAG", error=str(e))
                return []
                
    def add_documents_from_directory(self, directory_path: str, file_patterns: Optional[List[str]] = None) -> int:
        """Ingère tous les documents d'un répertoire"""
        if file_patterns is None:
            file_patterns = ["*.txt", "*.md", "*.py"]
            
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Répertoire inexistant: {directory_path}")
            
        docs_added = 0
        
        with self.telemetry.trace_operation("rag_bulk_ingest", {"directory": directory_path}):
            for pattern in file_patterns:
                for file_path in directory.glob(pattern):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        metadata = {
                            "filename": file_path.name,
                            "filepath": str(file_path),
                            "file_type": file_path.suffix
                        }
                        
                        self.add_document(content, metadata)
                        docs_added += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Erreur lecture fichier {file_path}: {e}")
                        
        self.logger.info(f"Ingestion terminée: {docs_added} documents ajoutés")
        return docs_added
        
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du système RAG"""
        try:
            if self.collection:
                count = self.collection.count()
            else:
                count = 0
                
            return {
                "document_count": count,
                "backend": "LlamaIndex+Chroma" if self.llama_index else "ChromaDB" if self.collection else "None",
                "embedding_model": self.embedding_model_name,
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            self.logger.error("Erreur récupération stats", error=str(e))
            return {"error": str(e)}
            
    def clear_all(self) -> bool:
        """Vide complètement le système RAG"""
        try:
            if self.collection:
                # Reset de la collection ChromaDB
                self.chroma_client.delete_collection(self.collection_name)
                self.collection = self.chroma_client.create_collection(
                    self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                
            self.logger.info("Système RAG vidé")
            return True
        except Exception as e:
            self.logger.error("Erreur vidage RAG", error=str(e))
            return False 