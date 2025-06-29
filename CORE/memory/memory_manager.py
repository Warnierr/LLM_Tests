from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
from datetime import datetime
import json
import logging
import uuid

class NinaMemoryManager:
    def __init__(
        self,
        db_path: str = "MEMORY/nina_memory.db",
        persist_directory: str = "MEMORY/chroma_db",
        collection_name: str = "conversations",
    ):
        """Gestionnaire de mémoire pour Nina avec RAG
        
        Args:
            db_path: Chemin vers la base de données SQLite
            persist_directory: Dossier où ChromaDB stocke les données
            collection_name: Nom de la collection ChromaDB à utiliser (par défaut « conversations »)
        """
        self.db_path = db_path
        self.setup_logging()
        
        # Initialisation de ChromaDB pour le stockage vectoriel
        self.chroma_client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False,
            )
        )
        
        # Création ou récupération de la collection pour les conversations
        try:
            self.conversation_collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Mémoire des conversations Nina"},
                get_or_create=True,
            )
        except TypeError:
            # Pour compatibilité avec versions antérieures de ChromaDB qui n'ont pas le paramètre get_or_create
            self.conversation_collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Mémoire des conversations Nina"},
            )
        
        # Modèle pour les embeddings
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        
        # Initialisation de la base SQLite
        self.init_sqlite_db()
        
    def setup_logging(self):
        """Configuration du système de logging"""
        logging.basicConfig(
            filename='LOGS/memory_manager.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('NinaMemoryManager')
        
    def init_sqlite_db(self):
        """Initialisation de la base SQLite avec les nouvelles tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_hierarchy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding_id TEXT,
                    memory_type TEXT NOT NULL,
                    importance_score FLOAT,
                    last_accessed DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
    def add_memory(self, content: str, memory_type: str = "conversation", 
                  metadata: Optional[Dict] = None) -> int:
        """Ajoute un nouveau souvenir dans le système
        
        Args:
            content: Contenu du souvenir
            memory_type: Type de mémoire (conversation, fact, preference)
            metadata: Métadonnées additionnelles
            
        Returns:
            ID du souvenir
        """
        try:
            # Création de l'embedding
            embedding = self.embedding_model.encode(content)
            
            # Génération d'un ID unique fiable (uuid4)
            memory_id = f"mem_{uuid.uuid4().hex}"
            
            # Préparation des métadonnées à transmettre à ChromaDB (doit être None ou dict non vide)
            chroma_metadata = metadata if metadata else None

            # Ajout dans ChromaDB
            add_kwargs = {
                "embeddings": [embedding.tolist()],
                "documents": [content],
                "ids": [memory_id],
            }
            if chroma_metadata is not None:
                add_kwargs["metadatas"] = [chroma_metadata]

            self.conversation_collection.add(**add_kwargs)
            
            # Ajout dans SQLite
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO memory_hierarchy 
                    (content, embedding_id, memory_type, importance_score, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (content, memory_id, memory_type, self._calculate_importance(content),
                     json.dumps(metadata or {}))
                )
                memory_id_sql = cursor.lastrowid
                
            self.logger.info(f"Nouveau souvenir ajouté: {memory_id_sql}")
            return int(memory_id_sql) if memory_id_sql is not None else -1
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout du souvenir: {str(e)}")
            raise
            
    def retrieve_relevant_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Récupère les souvenirs pertinents pour une requête
        
        Args:
            query: Requête pour la recherche
            limit: Nombre maximum de résultats
            
        Returns:
            Liste des souvenirs pertinents avec leurs scores
        """
        try:
            # Création de l'embedding de la requête
            query_embedding = self.embedding_model.encode(query)
            
            # Recherche dans ChromaDB
            results = self.conversation_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=limit
            )
            
            # Récupération des métadonnées depuis SQLite
            memories = []
            for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT * FROM memory_hierarchy WHERE embedding_id = ?",
                        (results['ids'][0][i],)
                    )
                    memory_data = cursor.fetchone()
                    
                    if memory_data:
                        memories.append({
                            'content': doc,
                            'relevance_score': 1 - distance,  # Conversion distance en score
                            'memory_type': memory_data[3],
                            'importance_score': memory_data[4],
                            'last_accessed': memory_data[5],
                            'metadata': json.loads(memory_data[7] or '{}')
                        })
                        
                        # Mise à jour de la date d'accès
                        conn.execute(
                            "UPDATE memory_hierarchy SET last_accessed = CURRENT_TIMESTAMP WHERE embedding_id = ?",
                            (results['ids'][0][i],)
                        )
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des souvenirs: {str(e)}")
            return []
            
    def _calculate_importance(self, content: str) -> float:
        """Calcule un score d'importance pour un contenu
        
        Pour l'instant, utilise une heuristique simple basée sur la longueur
        et la présence de mots-clés importants
        """
        importance_score = min(len(content.split()) / 100, 1.0)  # Basé sur la longueur
        
        # Mots-clés importants (à enrichir)
        important_keywords = ['préférence', 'important', 'rappelle', 'souviens', 
                            'crucial', 'vital', 'essentiel', 'objectif']
        
        for keyword in important_keywords:
            if keyword.lower() in content.lower():
                importance_score += 0.1
                
        return min(importance_score, 1.0)  # Normalisation entre 0 et 1 