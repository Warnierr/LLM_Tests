import os
from dotenv import load_dotenv
from groq import AsyncGroq
from typing import List, Dict, Optional, Union, cast, Any
from .memory.memory_manager import NinaMemoryManager
import json
import logging
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

# Sélection dynamique des backends LLM
import os
from dotenv import load_dotenv

# Backends possibles (imports optionnels)
from typing import List, Dict, Optional, Union, cast

try:
    from groq import AsyncGroq
except ImportError:  # groq n'est pas obligatoire si une autre clé est dispo
    AsyncGroq = None  # type: ignore

try:
    from openai import AsyncOpenAI  # type: ignore
except ImportError:
    AsyncOpenAI = None  # type: ignore

try:
    from anthropic import AsyncAnthropic  # type: ignore
except ImportError:
    AsyncAnthropic = None  # type: ignore

class Nina:
    def __init__(self):
        """Initialisation de Nina avec le nouveau système de mémoire"""
        load_dotenv()
        
        # Configuration du logging
        logging.basicConfig(
            filename='LOGS/nina.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('Nina')
        
        # Détection des backends disponibles
        self.backend = self._select_backend()

        # Initialisation des clients selon disponibilité
        self.clients: Dict[str, Any] = {}

        if self.backend == "groq":
            self.clients["groq"] = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))  # type: ignore[arg-type]
        elif self.backend == "openai":
            self.clients["openai"] = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # type: ignore[arg-type]
        elif self.backend == "anthropic":
            self.clients["anthropic"] = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))  # type: ignore[arg-type]
        
        # Initialisation du gestionnaire de mémoire
        self.memory_manager = NinaMemoryManager()
        
        # Configuration du système
        self.config = {
            'max_memory_items': 5,  # Nombre max de souvenirs à utiliser par conversation
            'memory_threshold': 0.6,  # Score minimum pour considérer un souvenir comme pertinent
            'llm_common': {
                'temperature': 0.7,
                'max_tokens': 800,
                'top_p': 0.9,
                'retry_attempts': 3,
                'retry_min_wait': 1,
                'retry_max_wait': 10,
            },
            'models': {
                'groq': "llama3-8b-8192",
                'openai': "gpt-4o-mini",  # peut être ajusté
                'anthropic': "claude-3-sonnet-20240229",
            },
        }
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _call_llm(self, context: str, user_message: str) -> str:
        """Appelle le LLM avec retry en cas d'erreur
        
        Args:
            context: Contexte pour le LLM
            user_message: Message de l'utilisateur
            
        Returns:
            Réponse du LLM
            
        Raises:
            Exception: En cas d'erreur après les retries
        """
        try:
            # Préparation des messages
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": user_message},
            ]

            model_name = self.config['models'][self.backend]
            params = dict(
                temperature=self.config['llm_common']['temperature'],
                max_tokens=self.config['llm_common']['max_tokens'],
                top_p=self.config['llm_common']['top_p'],
            )

            if self.backend == "groq":
                chat_completion = await self.clients["groq"].chat.completions.create(
                    messages=messages,
                    model=model_name,
                    **params,
                )
                content = chat_completion.choices[0].message.content  # type: ignore[attr-defined]

            elif self.backend == "openai":
                chat_completion = await self.clients["openai"].chat.completions.create(
                    messages=messages,
                    model=model_name,
                    **params,
                )
                content = chat_completion.choices[0].message.content  # type: ignore[attr-defined]

            elif self.backend == "anthropic":
                # Anthropic API utilise un paramètre system séparé
                chat_completion = await self.clients["anthropic"].messages.create(
                    system=context,
                    messages=[{"role": "user", "content": user_message}],
                    model=model_name,
                    max_tokens=self.config['llm_common']['max_tokens'],
                    temperature=self.config['llm_common']['temperature'],
                    top_p=self.config['llm_common']['top_p'],
                )
                content = chat_completion.content  # type: ignore[attr-defined]

            else:
                raise RuntimeError("Backend LLM inconnu")

            if not content:
                raise ValueError("Réponse du LLM vide")

            return cast(str, content)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'appel au LLM: {str(e)}")
            raise
        
    async def process_message(self, user_message: str) -> str:
        """Traite un message utilisateur avec le support de la mémoire
        
        Args:
            user_message: Message de l'utilisateur
            
        Returns:
            Réponse de Nina
        """
        try:
            # Récupération des souvenirs pertinents
            relevant_memories = self.memory_manager.retrieve_relevant_memories(
                user_message, 
                limit=self.config['max_memory_items']
            )
            
            # Filtrage des souvenirs par pertinence
            filtered_memories = [
                mem for mem in relevant_memories 
                if mem['relevance_score'] > self.config['memory_threshold']
            ]
            
            # Construction du contexte pour le LLM
            context = self._build_llm_context(filtered_memories)
            
            # Génération de la réponse via Groq avec retry
            response = await self._call_llm(context, user_message)
            
            # Sauvegarde de l'échange dans la mémoire
            self._save_conversation(user_message, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du message: {str(e)}")
            return "Désolé, j'ai rencontré une erreur. Pouvez-vous reformuler votre message ?"
            
    def _build_llm_context(self, memories: List[Dict]) -> str:
        """Construit le contexte pour le LLM en utilisant les souvenirs pertinents"""
        context = [
            "Tu es Nina, une assistante IA française intelligente avec une mémoire améliorée. "
            "Utilise les informations suivantes de ta mémoire pour personnaliser ta réponse:"
        ]
        
        if memories:
            context.append("\nSouvenirs pertinents:")
            for mem in memories:
                context.append(f"- {mem['content']} (Pertinence: {mem['relevance_score']:.2f})")
        else:
            context.append("\nAucun souvenir pertinent trouvé pour cette conversation.")
            
        return "\n".join(context)
        
    def _save_conversation(self, user_message: str, nina_response: str):
        """Sauvegarde l'échange dans la mémoire"""
        conversation = {
            "user": user_message,
            "nina": nina_response,
            "timestamp": "auto"  # Sera géré par la base de données
        }
        
        self.memory_manager.add_memory(
            content=json.dumps(conversation),
            memory_type="conversation",
            metadata={"type": "exchange"}
        ) 

    def _select_backend(self) -> str:
        """Détermine automatiquement quel backend utiliser en fonction
        des clés d'API disponibles. Priorité : Groq > OpenAI > Anthropic.
        """
        if os.getenv("GROQ_API_KEY") and AsyncGroq is not None:
            return "groq"
        if os.getenv("OPENAI_API_KEY") and AsyncOpenAI is not None:
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY") and AsyncAnthropic is not None:
            return "anthropic"
        raise ValueError(
            "Aucun backend LLM disponible : définissez GROQ_API_KEY, OPENAI_API_KEY ou ANTHROPIC_API_KEY."
        ) 