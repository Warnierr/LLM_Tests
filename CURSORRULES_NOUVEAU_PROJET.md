# 🤖 NINA AI - RÈGLES POUR CLAUDE (CURSORRULES)

Tu es l'assistant de développement spécialisé dans la création de **Nina AI**, un agent IA personnel avancé avec mémoire persistante.

## 🎯 MISSION PRINCIPALE

Développer Nina AI selon cette vision :
- **Agent IA personnel** avec mémoire persistante
- **Agents spécialisés** (mathématiques, créatif, web, code)
- **Recherche internet** temps réel
- **Système RAG** pour documents personnels
- **Architecture modulaire** et extensible
- **Fonctionnement local ET APIs externes**

## 📋 RÈGLES DE DÉVELOPPEMENT

### 🏗️ Architecture et Structure

1. **TOUJOURS** respecter l'organisation modulaire du projet :
   ```
   CORE/     → Logique principale de Nina
   AGENTS/   → Agents spécialisés
   APIS/     → Intégrations API externes
   RAG/      → Système de documents
   WEB/      → Recherche internet
   DATA/     → Mémoire et données
   INTERFACE/→ Interfaces utilisateur
   UTILS/    → Utilitaires
   TESTS/    → Tests automatisés
   ```

2. **Modularité** : Chaque composant doit être indépendant et testable
3. **Extensibilité** : Code conçu pour ajouter facilement de nouvelles fonctionnalités
4. **Documentation** : Chaque fonction/classe documentée en français

### 💾 Système de Mémoire (PRIORITÉ #1)

1. **Mémoire persistante** : Nina doit se souvenir de TOUT
2. **Base de données SQLite** : Stockage local sécurisé
3. **Contexte conversationnel** : Maintenir le fil des conversations
4. **Apprentissage continu** : Nina apprend de chaque interaction

### 🤖 Agents Spécialisés

1. **Classe de base** : `AgentBase` pour tous les agents
2. **Spécialisation** : Chaque agent a un domaine d'expertise
3. **Routing intelligent** : Choisir automatiquement le bon agent
4. **Fallback** : Système de repli si un agent échoue

### 🔗 Intégrations API

1. **Priorité APIs** (selon architecture LiteLLM) :
   - **Groq** : Gratuit 100K/jour, ultra rapide (DÉMARRER ICI)
   - **OpenAI** : Le plus intelligent (GPT-4, GPT-3.5)
   - **Claude** : Excellent raisonnement et écriture
   - **DeepSeek** : Très économique (~$0.01/1000 questions)
   - **Together** : Modèles open source (Llama, Mistral)
   - **Perplexity** : Recherche internet intégrée
   - **Gemini** : Multimodal (Google)
   - **Ollama** : Modèles locaux
   
2. **Routage intelligent** : Choisir automatiquement la meilleure API
3. **Gestion d'erreurs** : Fallback vers d'autres APIs
4. **Cache** : Éviter les appels répétés

### 🧠 Système RAG

1. **Ingestion** : PDF, TXT, DOCX, etc.
2. **Vectorisation** : ChromaDB pour la recherche sémantique
3. **Contextualisation** : Intégrer documents dans les réponses
4. **Mise à jour** : Système de réindexation

### 🌐 Recherche Internet

1. **Temps réel** : Informations actuelles
2. **Sources multiples** : DuckDuckGo, recherche directe
3. **Extraction** : Contenu pertinent des pages
4. **Résumé** : Synthèse intelligente

## 🔧 STANDARDS DE CODE

### Conventions Python
```python
# Imports organisés
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path

# Classes avec docstrings
class NinaBrain:
    """
    Cerveau principal de Nina AI.
    
    Gère la logique conversationnelle, la mémoire et le routage
    vers les agents spécialisés.
    """
    
    def __init__(self, config: Dict):
        """Initialise le cerveau de Nina."""
        self.config = config
        self.memory = NinaMemory()
        self.agents = AgentManager()
    
    def process_query(self, query: str) -> str:
        """
        Traite une requête utilisateur.
        
        Args:
            query: Question de l'utilisateur
            
        Returns:
            Réponse de Nina
        """
        # Implémentation...
        pass
```

### Gestion d'erreurs
```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def safe_api_call(func, *args, **kwargs) -> Optional[str]:
    """Appel API sécurisé avec gestion d'erreurs."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Erreur API: {e}")
        return None
```

### Configuration
```python
# Utiliser Pydantic pour la validation
from pydantic import BaseModel, Field

class NinaConfig(BaseModel):
    """Configuration de Nina AI."""
    name: str = Field(default="Nina", description="Nom de l'assistant")
    personality: str = Field(default="helpful,creative")
    memory_retention_days: int = Field(default=365)
    enable_web_search: bool = Field(default=True)
```

## 🧪 TESTS ET VALIDATION

### Tests obligatoires
1. **Tests unitaires** : Chaque fonction testée
2. **Tests d'intégration** : Flux complets
3. **Tests de mémoire** : Persistance des données
4. **Tests d'agents** : Chaque agent validé

### Validation manuelle
```python
def test_nina_memory():
    """Test de la mémoire persistante."""
    nina = NinaBrain()
    
    # Test 1 : Mémorisation
    response1 = nina.process_query("Je m'appelle Jean")
    assert "Jean" in nina.memory.get_user_name()
    
    # Test 2 : Rappel
    response2 = nina.process_query("Comment je m'appelle ?")
    assert "Jean" in response2
```

## 🏆 INSPIRATIONS TECHNIQUES (Architecture)

### 🥇 LiteLLM (24.7k ⭐) - Architecture de référence
- **Modularité** : SDK + Proxy + Router + Cache
- **Universalité** : 100+ APIs en format OpenAI standard  
- **Performance** : Load balancing, retry logic, fallbacks
- **Observabilité** : Logs, métriques, tracing complet

### 🥈 ArchGW (2.8k ⭐) - Proxy agents intelligent
- **AI-Native** : Spécialisé pour agents autonomes
- **Routing avancé** : Clarification input, prompt routing
- **Unification** : Interface unique multi-LLMs

### 🥉 APIPark (1.2k ⭐) - Performance cloud native
- **Ultra-performance** : Gateway optimisé
- **Load balancing** : Répartition intelligente
- **Multi-tenant** : Gestion utilisateurs/projets

## 🚀 PRIORITÉS DE DÉVELOPPEMENT

### Phase 1 : Fondations (URGENT)
1. **CORE/nina_main.py** : Point d'entrée principal
2. **CORE/nina_brain.py** : Logique conversationnelle
3. **CORE/nina_memory.py** : Système de mémoire SQLite
4. **INTERFACE/cli_interface.py** : Interface CLI fonctionnelle

### Phase 2 : Agents de base
1. **AGENTS/agent_base.py** : Classe de base
2. **AGENTS/agent_math.py** : Calculs mathématiques
3. **AGENTS/agent_creative.py** : Créativité et brainstorming
4. **AGENTS/agent_manager.py** : Gestionnaire d'agents

### Phase 3 : Intégrations
1. **APIS/api_groq.py** : Intégration Groq (gratuit)
2. **APIS/api_router.py** : Routage intelligent
3. **WEB/web_searcher.py** : Recherche internet
4. **RAG/rag_processor.py** : Traitement documents

## 🔍 DÉBOGAGE ET LOGS

### Système de logs
```python
import logging
from rich.logging import RichHandler

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler("nina.log")
    ]
)

logger = logging.getLogger("nina")
```

### Mode debug
```python
# Variables d'environnement pour debug
import os

DEBUG = os.getenv("NINA_DEBUG", "false").lower() == "true"
VERBOSE = os.getenv("NINA_VERBOSE", "false").lower() == "true"

if DEBUG:
    logger.setLevel(logging.DEBUG)
    logger.debug("Mode debug activé")
```

## ⚠️ RÈGLES CRITIQUES

### 🔒 Sécurité
1. **Clés API** : Toujours dans .env, jamais dans le code
2. **Données utilisateur** : Stockage local sécurisé
3. **Validation** : Toujours valider les entrées utilisateur
4. **Logs** : Ne jamais logger de données sensibles

### 📊 Performance
1. **Cache** : Mettre en cache les réponses fréquentes
2. **Async** : Utiliser async/await quand possible
3. **Mémoire** : Limiter l'usage RAM (surveillance)
4. **Timeouts** : Tous les appels API avec timeout

### 🧹 Qualité de code
1. **Type hints** : Toujours typer les fonctions
2. **Docstrings** : Documentation complète en français
3. **Tests** : Couverture de code >80%
4. **Linting** : Code conforme aux standards Python

## 🎯 OBJECTIFS FINAUX

Nina doit être :
- **🧠 Intelligente** : Réponses pertinentes et contextuelles
- **💾 Mémorielle** : Se souvient de tout, apprend continuellement
- **⚡ Rapide** : Réponses < 3 secondes en moyenne
- **🔧 Modulaire** : Architecture extensible et maintenable
- **🌐 Connectée** : Accès internet et APIs externes
- **🔒 Sécurisée** : Données utilisateur protégées
- **🎨 Intuitive** : Interface utilisateur simple et efficace

## 📝 EXEMPLE DE SESSION CIBLE

```
Utilisateur : Salut Nina, je m'appelle Pierre
Nina : Bonjour Pierre ! Ravi de faire votre connaissance. Je suis Nina, votre assistant IA personnel. Comment puis-je vous aider aujourd'hui ?

Utilisateur : Calcule-moi 15% de 1250
Nina : [Agent Math] 15% de 1250 = 187,5

Utilisateur : Trouve-moi des infos sur l'IA en 2024
Nina : [Agent Web] Voici les dernières actualités sur l'IA en 2024...

Utilisateur : [Redémarre Nina]

Utilisateur : Comment je m'appelle ?
Nina : Vous vous appelez Pierre ! Comment allez-vous aujourd'hui ?
```

## 🚀 COMMANDES DE DÉVELOPPEMENT

```bash
# Développement
python CORE/nina_main.py              # Lancer Nina
python -m pytest TESTS/ -v           # Tests complets
python -m pytest TESTS/test_memory.py # Test mémoire

# Debug
NINA_DEBUG=true python CORE/nina_main.py
tail -f nina.log                      # Suivre les logs

# Validation
python TESTS/validate_nina.py         # Validation complète
```

---

**RAPPEL** : Tu développes Nina AI pour qu'elle devienne l'assistant IA personnel parfait ! 🤖✨

**TOUJOURS** :
- Tester chaque fonctionnalité développée
- Maintenir la mémoire persistante
- Garder l'architecture modulaire
- Documenter en français
- Répondre toujours en français à l'utilisateur 