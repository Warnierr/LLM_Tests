"""
RAG simple sans ChromaDB pour éviter les conflits de dépendances
"""
import json
import os
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from ..utils.telemetry import get_telemetry


class SimpleRAGManager:
    """Gestionnaire RAG simple avec stockage JSON et embeddings locaux"""
    
    def __init__(
        self,
        persist_directory: str = "MEMORY/simple_rag",
        embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.documents_file = self.persist_directory / "documents.json"
        self.embeddings_file = self.persist_directory / "embeddings.npy"
        
        self.telemetry = get_telemetry()
        self.logger = self.telemetry.get_logger("SimpleRAG")
        
        # Modèle d'embedding
        self.logger.info("Chargement modèle embeddings")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Chargement des données existantes
        self.documents = self._load_documents()
        self.embeddings = self._load_embeddings()
        
        self.logger.info("SimpleRAG initialisé", document_count=len(self.documents))
        
    def _load_documents(self) -> List[Dict[str, Any]]:
        """Charge les documents depuis le fichier JSON"""
        if self.documents_file.exists():
            try:
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Erreur chargement documents: {e}")
        return []
        
    def _load_embeddings(self) -> Optional[np.ndarray]:
        """Charge les embeddings depuis le fichier numpy"""
        if self.embeddings_file.exists():
            try:
                return np.load(self.embeddings_file)
            except Exception as e:
                self.logger.warning(f"Erreur chargement embeddings: {e}")
        return None
        
    def _save_documents(self):
        """Sauvegarde les documents dans le fichier JSON"""
        try:
            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde documents: {e}")
            
    def _save_embeddings(self):
        """Sauvegarde les embeddings dans le fichier numpy"""
        if self.embeddings is not None:
            try:
                np.save(self.embeddings_file, self.embeddings)
            except Exception as e:
                self.logger.error(f"Erreur sauvegarde embeddings: {e}")
                
    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Ajoute un document au système RAG"""
        doc_id = f"doc_{uuid.uuid4().hex}"
        
        with self.telemetry.trace_operation("simple_rag_add", {"doc_id": doc_id}):
            try:
                # Créer l'embedding
                embedding = self.embedding_model.encode(text, normalize_embeddings=True)
                embedding = np.array(embedding)  # Conversion explicite en numpy array
                
                # Ajouter le document
                document = {
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata or {}
                }
                self.documents.append(document)
                
                # Ajouter l'embedding
                if self.embeddings is None:
                    self.embeddings = embedding.reshape(1, -1)
                else:
                    self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
                
                # Sauvegarder
                self._save_documents()
                self._save_embeddings()
                
                self.telemetry.record_memory_operation("simple_rag_add", 1)
                self.logger.info("Document ajouté", doc_id=doc_id, text_length=len(text))
                
                return doc_id
                
            except Exception as e:
                self.logger.error("Erreur ajout document", error=str(e))
                raise
                
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Recherche sémantique dans les documents"""
        
        with self.telemetry.trace_operation("simple_rag_search", {"query_length": len(query)}):
            try:
                if len(self.documents) == 0 or self.embeddings is None:
                    return []
                
                # Créer l'embedding de la requête
                query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
                
                # Calculer les similarités cosinus
                similarities = np.dot(self.embeddings, query_embedding)
                
                # Obtenir les indices triés par similarité décroissante
                sorted_indices = np.argsort(similarities)[::-1][:top_k]
                
                # Construire les résultats
                results = []
                for idx in sorted_indices:
                    doc = self.documents[idx]
                    score = float(similarities[idx])
                    
                    results.append({
                        "text": doc["text"],
                        "score": score,
                        "metadata": doc["metadata"],
                        "doc_id": doc["id"]
                    })
                
                self.telemetry.record_memory_operation("simple_rag_search", len(results))
                self.logger.info("Recherche effectuée", 
                               query_length=len(query), 
                               results_count=len(results))
                
                return results
                
            except Exception as e:
                self.logger.error("Erreur recherche", error=str(e))
                return []
                
    def add_documents_from_directory(self, directory_path: str, file_patterns: Optional[List[str]] = None) -> int:
        """Ingère tous les documents d'un répertoire"""
        if file_patterns is None:
            file_patterns = ["*.txt", "*.md", "*.py"]
            
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Répertoire inexistant: {directory_path}")
            
        docs_added = 0
        
        with self.telemetry.trace_operation("simple_rag_bulk_ingest", {"directory": directory_path}):
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
        return {
            "document_count": len(self.documents),
            "backend": "SimpleRAG",
            "embedding_model": "paraphrase-multilingual-mpnet-base-v2",
            "persist_directory": str(self.persist_directory),
            "total_size_mb": self._get_total_size()
        }
        
    def _get_total_size(self) -> float:
        """Calcule la taille totale du stockage en MB"""
        total_size = 0
        for file_path in [self.documents_file, self.embeddings_file]:
            if file_path.exists():
                total_size += file_path.stat().st_size
        return round(total_size / (1024 * 1024), 2)
        
    def clear_all(self) -> bool:
        """Vide complètement le système RAG"""
        try:
            self.documents = []
            self.embeddings = None
            
            # Supprimer les fichiers
            for file_path in [self.documents_file, self.embeddings_file]:
                if file_path.exists():
                    file_path.unlink()
                    
            self.logger.info("Système RAG vidé")
            return True
        except Exception as e:
            self.logger.error("Erreur vidage RAG", error=str(e))
            return False 