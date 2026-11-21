"""Interfaz común de backend y medición de una solicitud (DESARROLLO.md §2 y §3.3).

Toda la temporización usa `time.perf_counter()`; el reloj de pared (UTC) solo marca el inicio de cada
solicitud para alinearla después con la telemetría de GPU (fase 3).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Protocol

import httpx


class ErrorBackend(Exception):
    """El servidor respondió con un error (HTTP o dentro del stream). El mensaje es el del servidor."""


@dataclass(frozen=True)
class Solicitud:
    prompt: str
    max_tokens: int
    temperatura: float = 0.0
    semilla: int = 0


@dataclass(frozen=True)
class Fragmento:
    t: float                      # time.perf_counter() al recibirlo
    texto: str
    n_tokens: int | None = None   # tokens que trae el fragmento, si el backend lo informa
    final: dict | None = None     # último mensaje del servidor (Ollama: done=true con las métricas)


@dataclass(frozen=True)
class EstadoSalud:
    ok: bool
    detalle: str
    version: str | None = None


class Backend(Protocol):
    nombre: str

    async def salud(self) -> EstadoSalud: ...
    async def modelos(self) -> list[str]: ...
    async def info_modelo(self, modelo: str) -> dict: ...
    def generar(self, sol: Solicitud) -> AsyncIterator[Fragmento]: ...
    def metricas_servidor(self, final: dict | None) -> dict: ...


@dataclass
class Resultado:
    """Medida de una solicitud. Los campos coinciden con una entrada de `datos.json` (§5)."""

    t_inicio_utc: str
    ttft_s: float | None = None
    e2e_s: float | None = None
    tpot_s: float | None = None
    itl_s: list[float] = field(default_factory=list)
    n_fragmentos: int = 0
    tokens_por_fragmento: list[int] | None = None
    tokens_entrada: int | None = None     # los que informa el servidor, no los pedidos
    tokens_salida: int | None = None
    finish_reason: str | None = None
    prefill_tok_s_cliente: float | None = None
    decode_tok_s_cliente: float | None = None
    servidor: dict = field(default_factory=dict)
    error: str | None = None

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


async def medir(backend: Backend, sol: Solicitud) -> Resultado:
    """Lanza una solicitud en streaming y calcula TTFT, E2E, TPOT e ITL (§2).

    - TTFT: envío → primer fragmento con texto.
    - E2E: envío → último fragmento recibido (incluido el final sin texto).
    - TPOT: (E2E − TTFT) / (tokens_salida − 1), con los tokens que informa el servidor.
    - ITL: intervalos entre fragmentos con texto consecutivos (un fragmento puede traer > 1 token).

    Nunca lanza por un fallo de la solicitud: lo deja en `error` indicando dónde estaba (§6: si salta
    un tiempo de espera, se informa de dónde estaba la solicitud; no se reintenta).
    """
    res = Resultado(t_inicio_utc=_utc_ahora())
    marcas: list[float] = []
    n_tok: list[int | None] = []
    final: dict | None = None
    t0 = time.perf_counter()
    t_ultimo = None
    try:
        async for frag in backend.generar(sol):
            t_ultimo = frag.t
            res.n_fragmentos += 1
            if frag.texto:
                marcas.append(frag.t)
                n_tok.append(frag.n_tokens)
            if frag.final is not None:
                final = frag.final
    except httpx.TimeoutException as e:
        if not marcas:
            donde = "esperando el primer fragmento"
        else:
            donde = f"tras {len(marcas)} fragmentos con texto (último a {marcas[-1] - t0:.3f} s)"
        res.error = f"tiempo de espera agotado ({type(e).__name__}) {donde}"
    except (ErrorBackend, httpx.HTTPError) as e:
        res.error = f"{type(e).__name__}: {e}"

    if marcas:
        res.ttft_s = marcas[0] - t0
        res.itl_s = [b - a for a, b in zip(marcas, marcas[1:])]
    if t_ultimo is not None:
        res.e2e_s = t_ultimo - t0
    if all(n is not None for n in n_tok) and n_tok:
        res.tokens_por_fragmento = n_tok  # type: ignore[assignment]

    # metricas_servidor devuelve las métricas propias del backend más tres claves comunes
    # (tokens_entrada, tokens_salida, finish_reason), que suben al nivel de la solicitud.
    res.servidor = dict(backend.metricas_servidor(final))
    res.tokens_entrada = res.servidor.pop("tokens_entrada", None)
    res.tokens_salida = res.servidor.pop("tokens_salida", None)
    res.finish_reason = res.servidor.pop("finish_reason", None)
    if res.error is None and final is None:
        res.error = "el stream terminó sin mensaje final del servidor"

    if res.ttft_s and res.tokens_entrada:
        res.prefill_tok_s_cliente = res.tokens_entrada / res.ttft_s
    if res.ttft_s is not None and res.e2e_s is not None and res.tokens_salida and res.tokens_salida > 1:
        decode = res.e2e_s - res.ttft_s
        if decode > 0:
            res.tpot_s = decode / (res.tokens_salida - 1)
            res.decode_tok_s_cliente = (res.tokens_salida - 1) / decode
    return res
