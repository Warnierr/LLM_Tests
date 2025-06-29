# Architecture de Nina

## Vue d'ensemble
Nina suit une architecture modulaire orientée observabilité et évolutivité.

## Modules principaux

### CORE/nina_main.py
Point d'entrée principal avec :
- Gestion des backends LLM
- Orchestration intelligente
- Traitement des messages
- Télémétrie intégrée

### CORE/memory/
- `memory_manager.py` : Gestionnaire mémoire SQLite + ChromaDB
- `advanced_memory.py` : Système hiérarchique mémoire
- Scoring de pertinence dynamique

### CORE/utils/
- `telemetry.py` : OpenTelemetry + logs structurés
- `sentiment.py` : Analyse sentiment
- `agenda.py` : Gestion calendrier

### CORE/rag/
- `rag_manager.py` : Gestionnaire RAG unifié
- Fallback gracieux ChromaDB/LlamaIndex
- Ingestion documents automatisée

## Évolution prévue
1. **S0-T2** : RAG + ingestion docs ✅
2. **S0-T3** : Tests conversationnels
3. **S1-T1** : API recherche locale RAG
4. **S2-T1** : Refactorisation agents
5. **S3-T1** : Dashboard Grafana

## Patterns architecturaux
- **Dependency Injection** : Télémétrie injectable
- **Graceful Degradation** : Fallbacks multiples
- **Observer Pattern** : Métriques temps réel
- **Factory Pattern** : Création agents dynamique 