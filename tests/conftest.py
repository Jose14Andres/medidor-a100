import httpx
import pytest


@pytest.fixture(scope="session")
def cliente():
    """Un único cliente HTTP para toda la sesión: crear uno cuesta ~2 s en Windows (contexto SSL).

    trust_env=False: que un http_proxy del entorno no desvíe las solicitudes a 127.0.0.1.
    """
    with httpx.Client(trust_env=False, timeout=10) as c:
        yield c


DATOS = __import__("pathlib").Path(__file__).resolve().parent / "datos"


@pytest.fixture
def gpu_falsa(monkeypatch):
    """Sustituye nvidia-smi: los tests no dependen de que haya GPU (ni de que esté libre).

    Devuelve un dict mutable: `gpu_falsa["usado_mib"] = 0` simula una GPU vacía.
    """
    from medidor import conexion

    estado = {"usado_mib": 24819}

    def memoria():
        return [{"indice": 0, "nombre": "GPU falsa", "usado_mib": estado["usado_mib"],
                 "total_mib": 40960, "driver": "535.00"}]

    monkeypatch.setattr(conexion, "memoria_gpu", memoria)
    monkeypatch.setattr(conexion, "gpus_L", lambda: ["GPU 0: GPU falsa (UUID: GPU-falsa)"])
    return estado
