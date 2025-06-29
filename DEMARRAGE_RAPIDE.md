# 🚀 NINA AI - Démarrage Rapide

## ✨ Nouveau Projet Propre
Nina AI nouvelle génération avec **vrai agent IA proxy intelligent**.

### 🎯 Caractéristiques
- **Agent IA proxy** : Réponses intelligentes via APIs externes
- **Mémoire persistante** : Se souvient de vos conversations
- **Multi-APIs** : Groq (gratuit), OpenAI, Claude, etc.
- **Réponses concises** : Comme un vrai assistant IA
- **Architecture propre** : Code modulaire et professionnel

## 🚀 Installation Rapide

### 1. Dépendances
```bash
pip install requests python-dotenv
```

### 2. Configuration API (Groq gratuit recommandé)
```bash
# Créer le fichier .env
echo "GROQ_API_KEY=votre_cle_ici" > .env
```

🔗 **Obtenir clé Groq gratuite**: https://console.groq.com/keys

### 3. Démarrage
```bash
python start_nina.py
```

## 💬 Utilisation

```
🤖 NINA AI - Agent IA Personnel
💬 Conversation intelligente avec mémoire
⚡ Tapez vos questions naturellement
==================================================

👤 Vous: salut nina
🤖 Nina: Salut ! Comment puis-je t'aider aujourd'hui ?

👤 Vous: comment ça va ?
🤖 Nina: Ça va bien merci ! Et toi, comment tu te sens ?

👤 Vous: stats
📊 STATISTIQUES NINA AI
==============================
💬 Conversations: 2
⚡ Temps moyen: 1.2s
🔗 APIs: groq
💾 Mémoire: Activée
```

## 🔧 Configuration Avancée

### APIs Multiples
```env
# .env - Plusieurs APIs disponibles
GROQ_API_KEY=gsk_...        # Gratuit, rapide
OPENAI_API_KEY=sk-...       # Payant, précis
ANTHROPIC_API_KEY=sk-...    # Payant, intelligent
```

Nina choisira automatiquement la meilleure API disponible.

### Commandes Spéciales
- `stats` - Afficher statistiques
- `debug [question]` - Infos techniques
- `exit` - Quitter

## 📁 Structure Projet
```
Nina AI/
├── start_nina.py           # 🚀 Script démarrage
├── CORE/
│   ├── nina_main.py        # 🤖 Agent principal
│   ├── config_example.env  # ⚙️ Config exemple
│   └── requirements.txt    # 📦 Dépendances
├── MEMORY/                 # 💾 Base données
├── LOGS/                   # 📝 Journaux
└── README_NOUVEAU_PROJET.md # 📖 Doc complète
```

## 🎯 Philosophie
- **Intelligence réelle** : Pas de réponses pré-codées
- **Proxy agent IA** : Utilise de vrais LLMs externes
- **Simplicité** : Architecture claire et modulaire
- **Performance** : Réponses rapides et pertinentes

## 🆘 Dépannage

### Erreur "Aucune API disponible"
1. Vérifiez le fichier `.env`
2. Obtenez une clé Groq gratuite
3. Testez la connexion internet

### Réponses lentes
- Groq est plus rapide qu'OpenAI
- Vérifiez votre connexion
- Les APIs externes peuvent avoir des quotas

## 🔥 Prochaines Étapes
1. **Test** : `python start_nina.py`
2. **Configuration** : Ajoutez vos APIs
3. **Personnalisation** : Modifiez les prompts
4. **Extensions** : Ajoutez de nouvelles fonctionnalités

**Nina AI - Agent IA vraiment intelligent ! 🧠✨** 