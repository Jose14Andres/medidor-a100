import httpx
import pytest


@pytest.fixture(scope="session")
def cliente():
    """Un único cliente HTTP para toda la sesión: crear uno cuesta ~2 s en Windows (contexto SSL).

    trust_env=False: que un http_proxy del entorno no desvíe las solicitudes a 127.0.0.1.
    """
    with httpx.Client(trust_env=False, timeout=10) as c:
        yield c
