import pytest
import re
from CORE.nina_main import Nina

pytestmark = pytest.mark.asyncio

async def test_salutation_once():
    nina = Nina()
    # premier message => peut contenir un bonjour long
    resp1 = await nina.process_message("Salut Nina !")
    # second message quelconque
    resp2 = await nina.process_message("Comment vas-tu ?")
    # la deuxième réponse ne doit pas contenir la longue salutation répétée
    assert resp2.lower().count("bonjour") <= 1

async def test_web_search():
    nina = Nina()
    resp = await nina.process_message("Peux-tu chercher OpenAI sur internet ?")
    # Doit contenir une URL http et du texte descriptif (>20 caractères)
    assert re.search(r"https?://", resp)
    assert len(resp) > 40 