import asyncio
import time
import pytest

from CORE.nina_main import Nina

pytestmark = pytest.mark.asyncio


async def _run_conversation(nina: Nina, prompts):
    """Helper to feed prompts sequentially and return list of responses."""
    responses = []
    for p in prompts:
        start = time.perf_counter()
        resp = await nina.process_message(p)
        latency = time.perf_counter() - start
        responses.append((p, resp, latency))
    return responses


async def test_short_term_memory():
    """Vérifie que Nina se souvient du prénom et de la boisson préférée."""
    nina = Nina()

    prompts = [
        "Je m'appelle Bob.",
        "J'adore le thé.",
        "Comment je m'appelle ?",
        "Quelle boisson je préfère ?",
    ]
    responses = await _run_conversation(nina, prompts)

    # Les deux dernières réponses doivent contenir les infos antérieures
    assert "Bob".lower() in responses[2][1].lower()
    assert "thé" in responses[3][1].lower()


async def test_mutation_memory():
    """Vérifie qu'une information peut être mise à jour."""
    nina = Nina()
    await nina.process_message("Ma couleur préférée est le bleu.")
    await nina.process_message("Finalement, ma couleur préférée est le rouge.")
    resp = await nina.process_message("Quelle est ma couleur préférée ?")
    assert "rouge" in resp.lower()
    assert "bleu" not in resp.lower()


async def test_long_session_recall():
    """Teste 30 tours de conversation aléatoire puis rappel."""
    nina = Nina()
    # Inject fact early
    await nina.process_message("Rappelle-toi que j\'ai un chat nommé Pixel.")
    # Noise conversation
    for i in range(30):
        await nina.process_message(f"Parlons d'autre chose {i}.")
    recall = await nina.process_message("Comment s'appelle mon chat ?")
    assert "pixel" in recall.lower() 