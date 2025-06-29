"""Tests pour le système RAG de Nina"""
import pytest
import tempfile
import shutil
from pathlib import Path
from CORE.rag.simple_rag import SimpleRAGManager


class TestSimpleRAG:
    """Tests pour SimpleRAGManager"""
    
    def setup_method(self):
        """Setup pour chaque test"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.rag = SimpleRAGManager(persist_directory=str(self.temp_dir))
        
    def teardown_method(self):
        """Cleanup après chaque test"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
    def test_initialization(self):
        """Test que le RAG s'initialise correctement"""
        assert self.rag is not None
        assert len(self.rag.documents) == 0
        assert self.rag.embeddings is None
        
        stats = self.rag.get_stats()
        assert stats["backend"] == "SimpleRAG"
        assert stats["document_count"] == 0
        
    def test_add_document(self):
        """Test ajout d'un document"""
        text = "Ceci est un document de test pour Nina"
        metadata = {"type": "test", "author": "pytest"}
        
        doc_id = self.rag.add_document(text, metadata)
        
        assert doc_id is not None
        assert doc_id.startswith("doc_")
        assert len(self.rag.documents) == 1
        assert self.rag.embeddings is not None
        assert self.rag.embeddings.shape[0] == 1
        
        # Vérifier le contenu
        doc = self.rag.documents[0]
        assert doc["text"] == text
        assert doc["metadata"] == metadata
        assert doc["id"] == doc_id
        
    def test_search_empty(self):
        """Test recherche dans un RAG vide"""
        results = self.rag.search("test query")
        assert results == []
        
    def test_search_with_documents(self):
        """Test recherche avec des documents"""
        # Ajouter quelques documents
        docs = [
            ("Nina est une assistante IA française", {"type": "info"}),
            ("OpenTelemetry permet la télémétrie", {"type": "tech"}),
            ("Python est un langage de programmation", {"type": "prog"})
        ]
        
        for text, metadata in docs:
            self.rag.add_document(text, metadata)
            
        # Test recherche
        results = self.rag.search("assistante française", top_k=2)
        
        assert len(results) <= 2
        assert len(results) > 0
        
        # Le premier résultat devrait être le plus pertinent
        top_result = results[0]
        assert "score" in top_result
        assert "text" in top_result
        assert "metadata" in top_result
        assert "doc_id" in top_result
        assert top_result["score"] > 0
        
        # Vérifier que le score le plus élevé est en premier
        if len(results) > 1:
            assert results[0]["score"] >= results[1]["score"]
            
    def test_add_documents_from_directory_nonexistent(self):
        """Test ingestion d'un répertoire inexistant"""
        with pytest.raises(ValueError, match="Répertoire inexistant"):
            self.rag.add_documents_from_directory("/chemin/inexistant")
            
    def test_add_documents_from_directory(self):
        """Test ingestion d'un répertoire"""
        # Créer un répertoire temporaire avec des fichiers
        test_dir = self.temp_dir / "test_docs"
        test_dir.mkdir()
        
        # Créer quelques fichiers de test
        (test_dir / "doc1.txt").write_text("Contenu du document 1", encoding='utf-8')
        (test_dir / "doc2.md").write_text("# Contenu markdown\nTexte du document 2", encoding='utf-8')
        (test_dir / "ignore.log").write_text("Ce fichier sera ignoré", encoding='utf-8')
        
        # Ingérer les documents
        count = self.rag.add_documents_from_directory(str(test_dir), ["*.txt", "*.md"])
        
        assert count == 2
        assert len(self.rag.documents) == 2
        
        # Vérifier les métadonnées
        filenames = [doc["metadata"]["filename"] for doc in self.rag.documents]
        assert "doc1.txt" in filenames
        assert "doc2.md" in filenames
        
    def test_persistence(self):
        """Test de la persistance des données"""
        # Ajouter un document
        text = "Document persistant"
        doc_id = self.rag.add_document(text, {"persistent": True})
        
        # Créer une nouvelle instance avec le même répertoire
        rag2 = SimpleRAGManager(persist_directory=str(self.temp_dir))
        
        # Vérifier que les données sont récupérées
        assert len(rag2.documents) == 1
        assert rag2.documents[0]["text"] == text
        assert rag2.documents[0]["id"] == doc_id
        assert rag2.embeddings is not None
        assert rag2.embeddings.shape[0] == 1
        
    def test_clear_all(self):
        """Test du vidage complet"""
        # Ajouter quelques documents
        self.rag.add_document("Document 1", {})
        self.rag.add_document("Document 2", {})
        
        assert len(self.rag.documents) == 2
        assert self.rag.embeddings is not None
        
        # Vider
        success = self.rag.clear_all()
        
        assert success is True
        assert len(self.rag.documents) == 0
        assert self.rag.embeddings is None
        
        # Vérifier les stats
        stats = self.rag.get_stats()
        assert stats["document_count"] == 0
        
    def test_get_stats(self):
        """Test des statistiques"""
        # Stats initiales
        stats = self.rag.get_stats()
        assert stats["document_count"] == 0
        assert stats["backend"] == "SimpleRAG"
        assert "embedding_model" in stats
        assert "persist_directory" in stats
        assert "total_size_mb" in stats
        
        # Ajouter un document et revérifier
        self.rag.add_document("Test document", {})
        
        stats = self.rag.get_stats()
        assert stats["document_count"] == 1
        assert stats["total_size_mb"] >= 0
        
    def test_search_top_k(self):
        """Test du paramètre top_k dans la recherche"""
        # Ajouter plusieurs documents
        for i in range(10):
            self.rag.add_document(f"Document numéro {i} avec du contenu", {"num": i})
            
        # Test avec différentes valeurs de top_k
        results_3 = self.rag.search("document", top_k=3)
        results_5 = self.rag.search("document", top_k=5)
        results_20 = self.rag.search("document", top_k=20)  # Plus que disponible
        
        assert len(results_3) == 3
        assert len(results_5) == 5
        assert len(results_20) == 10  # Maximum disponible
        
    def test_error_handling(self):
        """Test de la gestion d'erreurs"""
        # Le RAG devrait gérer gracieusement les erreurs
        
        # Test avec du texte vide
        doc_id = self.rag.add_document("", {})
        assert doc_id is not None
        
        # Test recherche avec query vide
        results = self.rag.search("", top_k=5)
        assert isinstance(results, list)  # Ne devrait pas planter


if __name__ == "__main__":
    pytest.main([__file__]) 