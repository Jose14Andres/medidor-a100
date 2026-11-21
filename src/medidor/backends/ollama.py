"""Backend Ollama (API nativa, Ollama 0.24). DESARROLLO.md §2, §3.3 y §12.2.

Se mide con `POST /api/generate` en modo `raw` (sin plantilla de chat), comparable con
`/v1/completions` de vLLM y con la tanda del job 36268 de `medir_ollama.py`. El stream es NDJSON:
un objeto por línea; el último trae `done: true` y las métricas del servidor en nanosegundos.

Notas de §2 que se aplican aquí:
- `options.num_ctx` se fija siempre: sin él, Ollama trunca el prompt en silencio.
- `keep_alive` se fija siempre: sin él, el modelo se descarga y `load_duration` contamina el TTFT.
- Ollama 0.24 no tiene `ignore_eos`: la salida puede acabar antes; se guarda `done_reason`.
- Ollama no informa de cuántos tokens trae cada línea del stream (`n_tokens = None`).
"""

from __future__ import annotations

import json
import time
from typing import AsyncIterator

import httpx

from medidor.backends.base import ErrorBackend, EstadoSalud, Fragmento, Solicitud

NS = 1e9
TIMEOUT_CONEXION_S = 10.0


def _s(final: dict, clave: str) -> float | None:
    v = final.get(clave)
    return None if v is None else v / NS


class BackendOllama:
    nombre = "ollama"

    def __init__(self, url: str, modelo: str, num_ctx: int, keep_alive: str, timeout_s: float) -> None:
        self.url = url.rstrip("/")
        self.modelo = modelo
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.timeout_s = timeout_s
        # trust_env=False: que un http_proxy del entorno no desvíe las solicitudes a 127.0.0.1.
        # El tiempo de espera es por operación (sin datos durante timeout_s), no total (§6).
        self._cliente = httpx.AsyncClient(
            base_url=self.url, trust_env=False,
            timeout=httpx.Timeout(timeout_s, connect=TIMEOUT_CONEXION_S))

    async def __aenter__(self) -> "BackendOllama":
        return self

    async def __aexit__(self, *exc) -> None:
        await self._cliente.aclose()

    # --- consultas ---------------------------------------------------------------------------

    async def _get(self, ruta: str, timeout: float = TIMEOUT_CONEXION_S) -> httpx.Response:
        return await self._cliente.get(ruta, timeout=timeout)

    async def salud(self) -> EstadoSalud:
        try:
            r = await self._get("/api/version")
        except httpx.HTTPError as e:
            return EstadoSalud(False, f"{self.url}/api/version no responde: {type(e).__name__}: {e}")
        if r.status_code != 200:
            return EstadoSalud(False, f"{self.url}/api/version respondió {r.status_code}")
        version = r.json().get("version")
        return EstadoSalud(True, f"{self.url}/api/version → 200 (Ollama {version})", version)

    async def modelos(self) -> list[str]:
        r = await self._get("/api/tags")
        r.raise_for_status()
        return [m.get("name") or m.get("model") for m in r.json().get("models", [])]

    async def modelos_cargados(self) -> list[dict]:
        """`/api/ps`: modelos en memoria con `size` y `size_vram` (bytes)."""
        r = await self._get("/api/ps")
        r.raise_for_status()
        return r.json().get("models", [])

    async def info_modelo(self, modelo: str) -> dict:
        r = await self._cliente.post("/api/show", json={"model": modelo}, timeout=TIMEOUT_CONEXION_S * 3)
        r.raise_for_status()
        d = r.json()
        info = d.get("model_info") or {}
        ctx = next((v for k, v in info.items() if k.endswith(".context_length")), None)
        return {"detalles": d.get("details"), "context_length": ctx}

    # --- generación --------------------------------------------------------------------------

    def cuerpo(self, sol: Solicitud) -> dict:
        return {
            "model": self.modelo, "prompt": sol.prompt, "raw": True, "stream": True,
            "keep_alive": self.keep_alive,
            "options": {"num_predict": sol.max_tokens, "num_ctx": self.num_ctx,
                        "temperature": sol.temperatura, "seed": sol.semilla},
        }

    async def generar(self, sol: Solicitud) -> AsyncIterator[Fragmento]:
        async with self._cliente.stream("POST", "/api/generate", json=self.cuerpo(sol)) as r:
            if r.status_code != 200:
                cuerpo = (await r.aread()).decode(errors="replace")
                try:
                    cuerpo = json.loads(cuerpo).get("error", cuerpo)
                except (json.JSONDecodeError, AttributeError):
                    pass
                raise ErrorBackend(f"HTTP {r.status_code}: {cuerpo}")
            async for linea in r.aiter_lines():
                t = time.perf_counter()
                if not linea.strip():
                    continue
                try:
                    d = json.loads(linea)
                except json.JSONDecodeError as e:
                    raise ErrorBackend(f"línea NDJSON inválida: {linea[:200]!r}") from e
                if d.get("error"):
                    raise ErrorBackend(str(d["error"]))
                yield Fragmento(t=t, texto=d.get("response") or "",
                                final=d if d.get("done") else None)

    def metricas_servidor(self, final: dict | None) -> dict:
        if not final:
            return {}
        m = {
            "prompt_eval_count": final.get("prompt_eval_count"),
            "prompt_eval_duration_s": _s(final, "prompt_eval_duration"),
            "eval_count": final.get("eval_count"),
            "eval_duration_s": _s(final, "eval_duration"),
            "load_duration_s": _s(final, "load_duration"),
            "total_duration_s": _s(final, "total_duration"),
            "done_reason": final.get("done_reason"),
        }
        # Mismas definiciones que medir_ollama.py, para poder comparar con el job 36268.
        if m["prompt_eval_count"] and m["prompt_eval_duration_s"]:
            m["prefill_tok_s"] = m["prompt_eval_count"] / m["prompt_eval_duration_s"]
        if m["eval_count"] and m["eval_duration_s"]:
            m["decode_tok_s"] = m["eval_count"] / m["eval_duration_s"]
        m["tokens_entrada"] = m["prompt_eval_count"]
        m["tokens_salida"] = m["eval_count"]
        m["finish_reason"] = m["done_reason"]
        return m
