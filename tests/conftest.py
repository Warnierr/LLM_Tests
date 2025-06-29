from pathlib import Path
import sys

# Assure que le dossier racine du projet est dans sys.path pour les imports absolus (e.g. CORE)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR)) 