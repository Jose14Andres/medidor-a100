import asyncio
import socket

import pytest

from medidor.backends.base import Solicitud, medir
from medidor.backends.ollama import BackendOllama
from servidor_falso import Ajustes, arrancar

PROMPT_8 = "a b c d e f g h"


def _medir(url, sol, modelo="m", num_ctx=4096, timeout_s=10):
    async def corrida():
        async with BackendOllama(url, modelo, num_ctx, "30m", timeout_s) as be:
            return await medir(be, sol)
    return asyncio.run(corrida())


def test_metricas_contra_latencias_fijas():
    aj = Ajustes(modelo="m", ttft_s=0.05, itl_s=0.01)
    with arrancar(aj) as url:
        r = _medir(url, Solicitud(PROMPT_8, 20))
    assert r.error is None
    assert r.tokens_entrada == 8 and r.tokens_salida == 20 and r.finish_reason == "length"
    assert r.n_fragmentos == 21                              # 20 con texto + el final
    assert len(r.itl_s) == 19
    assert 0.05 <= r.ttft_s < 0.5
    assert r.tpot_s == pytest.approx(0.01, abs=0.005)
    assert all(x == pytest.approx(0.01, abs=0.01) for x in r.itl_s)
    assert r.servidor["prefill_tok_s"] == pytest.approx(8 / 0.05)
    assert r.servidor["decode_tok_s"] == pytest.approx(20 / (0.01 * 19))
    assert r.prefill_tok_s_cliente == pytest.approx(8 / r.ttft_s)
    assert "tokens_entrada" not in r.servidor                # subidas al nivel de la solicitud
    assert r.tokens_por_fragmento is None                    # Ollama no lo informa
    assert r.t_inicio_utc.endswith("Z")


def test_cuerpo_fija_raw_num_ctx_y_keep_alive():
    be = BackendOllama("http://127.0.0.1:1", "m", 8192, "30m", 5)
    c = be.cuerpo(Solicitud("x", 128, 0.0, 1234))
    assert c["raw"] is True and c["stream"] is True and c["keep_alive"] == "30m"
    assert c["options"] == {"num_predict": 128, "num_ctx": 8192, "temperature": 0.0, "seed": 1234}
    asyncio.run(be.__aexit__(None, None, None))


def test_modelo_inexistente():
    with arrancar(Ajustes(modelo="m")) as url:
        r = _medir(url, Solicitud(PROMPT_8, 4), modelo="otro")
    assert r.error and "HTTP 404" in r.error and "not found" in r.error


def test_timeout_dice_donde_estaba():
    with arrancar(Ajustes(modelo="m", ttft_s=1.0)) as url:
        r = _medir(url, Solicitud(PROMPT_8, 4), timeout_s=0.2)
    assert r.error and "tiempo de espera agotado" in r.error and "esperando el primer fragmento" in r.error
    assert r.ttft_s is None


def test_servidor_caido():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        puerto = s.getsockname()[1]
    r = _medir(f"http://127.0.0.1:{puerto}", Solicitud(PROMPT_8, 4))
    assert r.error and "ConnectError" in r.error
