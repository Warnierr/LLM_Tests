import asyncio
import os
from CORE.nina_main import Nina
import logging
from datetime import datetime
import traceback
from dotenv import load_dotenv

async def main():
    """Point d'entrée principal de Nina"""
    # Configuration des dossiers nécessaires
    os.makedirs("LOGS", exist_ok=True)
    os.makedirs("MEMORY", exist_ok=True)
    os.makedirs("MEMORY/chroma_db", exist_ok=True)
    
    # Configuration du logging
    log_file = f'LOGS/nina_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('NinaMain')
    
    # Ajout d'un handler pour la console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    try:
        # Chargement explicite du fichier .env
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if not os.path.exists(env_path):
            logger.error(f"Fichier .env non trouvé à l'emplacement : {env_path}")
            print("\n❌ Erreur: Fichier .env non trouvé")
            print("➡️ Créez un fichier .env avec votre clé API Groq")
            print("🔗 Obtenir une clé gratuite: https://console.groq.com/keys")
            return
            
        load_dotenv(env_path)
        logger.debug(f"Fichier .env chargé depuis : {env_path}")
        
        # Vérification de la clé API
        if not os.getenv('GROQ_API_KEY'):
            logger.error("Clé API GROQ_API_KEY non trouvée dans .env")
            print("\n❌ Erreur: Clé API GROQ_API_KEY non trouvée dans .env")
            print("➡️ Ajoutez GROQ_API_KEY=votre_clé dans le fichier .env")
            print("🔗 Obtenir une clé gratuite: https://console.groq.com/keys")
            return
            
        # Initialisation de Nina
        nina = Nina()
        logger.info("Nina initialisée avec succès")
        
        print("\n🤖 NINA AI - Assistant Personnel Intelligent")
        print("💭 Mémoire améliorée activée")
        print("📝 Tapez 'exit' pour quitter")
        print("=" * 50 + "\n")
        
        # Boucle principale
        while True:
            try:
                user_input = input("👤 Vous: ").strip()
                
                if user_input.lower() == 'exit':
                    print("\n👋 Au revoir !")
                    break
                    
                if not user_input:
                    continue
                    
                # Traitement du message
                response = await nina.process_message(user_input)
                print(f"\n🤖 Nina: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Au revoir !")
                break
            except Exception as e:
                logger.error(f"Erreur dans la boucle principale: {str(e)}")
                print("\n❌ Désolé, une erreur est survenue.")
                continue
                
    except Exception as e:
        logger.error(f"Erreur fatale: {str(e)}\n{traceback.format_exc()}")
        print("\n❌ Erreur critique lors du démarrage de Nina")
        print("📝 Consultez les logs pour plus de détails:", log_file)
        
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Au revoir !")
    except Exception as e:
        print(f"\n❌ Erreur fatale: {str(e)}") 