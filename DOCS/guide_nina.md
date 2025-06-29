# Guide d'utilisation de Nina

## Introduction
Nina est une assistante IA française dotée d'une mémoire avancée et de capacités d'orchestration multi-agents.

## Fonctionnalités principales

### Mémoire intelligente
- Mémoire vectorielle avec ChromaDB
- Récupération contextuelle des conversations passées
- Apprentissage continu des préférences utilisateur

### Orchestration multi-backend
- Support Groq, OpenAI, Anthropic
- Sélection automatique du meilleur backend
- Optimisation des performances par agent

### Système de télémétrie
- Tracing distribué avec OpenTelemetry
- Métriques de performance et d'utilisation
- Logs structurés JSON

### Capacités système
- Exécution sécurisée de commandes
- Gestion de fichiers avec sandbox
- Intégration agenda et préférences

## Installation
```bash
pip install -r requirements.txt
python start_nina.py
```

## Configuration
Créer un fichier `.env` avec :
```
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

## Exemples d'utilisation
- "Salut Nina, comment ça va ?"
- "Recherche des informations sur Python"
- "Écris un fichier hello.py avec un Hello World"
- "Agenda demain à 14h réunion projet" 