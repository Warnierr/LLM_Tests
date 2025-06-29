import pytest
import os
import json
from CORE.memory.memory_manager import NinaMemoryManager
import shutil
import asyncio
from CORE.nina_main import Nina

@pytest.fixture
def memory_manager():
    """Fixture pour créer un gestionnaire de mémoire de test"""
    # Utilisation d'une base de données temporaire pour les tests
    test_db_path = "MEMORY/test_memory.db"
    test_chroma_path = "MEMORY/test_chroma_db"
    
    # Nettoyage des fichiers de test précédents
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    if os.path.exists(test_chroma_path):
        shutil.rmtree(test_chroma_path)
        
    manager = NinaMemoryManager(db_path=test_db_path)
    yield manager
    
    # Nettoyage après les tests
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    if os.path.exists(test_chroma_path):
        shutil.rmtree(test_chroma_path)

def test_add_memory(memory_manager):
    """Test l'ajout d'un souvenir"""
    content = "Ceci est un test de mémoire"
    memory_id = memory_manager.add_memory(content)
    
    assert memory_id is not None
    assert isinstance(memory_id, int)
    
def test_retrieve_memories(memory_manager):
    """Test la récupération des souvenirs"""
    # Ajout de plusieurs souvenirs
    contents = [
        "J'aime beaucoup le chocolat",
        "Ma couleur préférée est le bleu",
        "Je déteste les épinards",
        "Le ciel est bleu aujourd'hui"
    ]
    
    for content in contents:
        memory_manager.add_memory(content)
        
    # Test de récupération avec une requête pertinente
    memories = memory_manager.retrieve_relevant_memories("Quelle est ma couleur préférée ?")
    
    assert len(memories) > 0
    assert any("bleu" in mem['content'].lower() for mem in memories)
    
def test_importance_scoring(memory_manager):
    """Test le calcul du score d'importance"""
    # Test avec un contenu important
    important_content = "C'est très important de se rappeler que je suis allergique aux arachides"
    memory_id = memory_manager.add_memory(important_content)
    
    # Test avec un contenu normal
    normal_content = "Il fait beau aujourd'hui"
    normal_memory_id = memory_manager.add_memory(normal_content)
    
    # Récupération des deux souvenirs
    memories = memory_manager.retrieve_relevant_memories("important allergies")
    
    # Vérification que le contenu important a un score plus élevé
    important_memory = next(mem for mem in memories if "allergique" in mem['content'])
    normal_memory = next((mem for mem in memories if "beau" in mem['content']), None)
    
    if normal_memory:
        assert important_memory['importance_score'] > normal_memory['importance_score']

@pytest.mark.asyncio
async def test_nina_integration():
    """Test l'intégration complète avec Nina"""
    nina = Nina()
    
    # Test d'une conversation simple
    response = await nina.process_message("Bonjour, comment vas-tu ?")
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Test de la mémoire à court terme
    await nina.process_message("Je m'appelle Alice")
    response = await nina.process_message("Comment je m'appelle ?")
    assert "Alice" in response.lower()
    
    # Test de la pertinence des souvenirs
    await nina.process_message("Ma couleur préférée est le rouge")
    await nina.process_message("J'aime beaucoup les chats")
    response = await nina.process_message("Quelle est ma couleur préférée ?")
    assert "rouge" in response.lower()

def test_memory_persistence(memory_manager):
    """Test la persistance des données"""
    # Ajout d'un souvenir
    content = "Information importante à retenir"
    memory_manager.add_memory(content)
    
    # Création d'un nouveau gestionnaire de mémoire avec la même base
    new_manager = NinaMemoryManager(db_path=memory_manager.db_path)
    
    # Vérification que le souvenir est toujours accessible
    memories = new_manager.retrieve_relevant_memories("information importante")
    assert len(memories) > 0
    assert content in memories[0]['content']

if __name__ == "__main__":
    pytest.main([__file__]) 