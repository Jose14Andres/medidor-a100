"""Prompts de N tokens con prefijo único (DESARROLLO.md §2 y §12.1).

Ollama 0.24 no tiene endpoint de tokenización, así que el prompt se construye en *palabras* y se
calibra contra el propio servidor: una solicitud de sondeo (`num_predict = 1`) devuelve
`prompt_eval_count`, el número real de tokens de entrada. Con dos o más sondeos se ajusta el número de
palabras hasta acercarse a los tokens pedidos (§12.1, decisión del 20-nov-2026).

Cada prompt empieza por un prefijo único (`Solicitud <id>-<nonce>.`) para que el servidor no reutilice
la caché KV de la solicitud anterior; si no, el prefill de las repeticiones sale artificialmente corto.
El id y el nonce tienen ancho fijo para que el prefijo tokenice siempre parecido.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable

# El mismo texto de relleno que ollama/medir_ollama.py, para que la validación contra el job 36268
# compare prompts equivalentes.
TEXTO = ("El rendimiento de la inferencia de modelos de lenguaje depende de la fase de prefill, "
         "que procesa el contexto completo en paralelo, y de la fase de decode, que genera un token "
         "tras otro y suele estar limitada por el ancho de banda de memoria. ")
_PALABRAS = TEXTO.split()

TOLERANCIA_ENTRADA = 0.02   # G7: tokens reales dentro de ±2 % de los pedidos
MAX_SONDEOS = 8


def cuerpo(n_palabras: int) -> str:
    return " ".join(_PALABRAS[k % len(_PALABRAS)] for k in range(n_palabras))


class GeneradorPrompts:
    """Prompts reproducibles: mismo `semilla` → mismos prompts, pero cada id con prefijo distinto."""

    def __init__(self, semilla: int) -> None:
        self._rng = random.Random(semilla)
        self._siguiente = 0

    def nuevo(self, n_palabras: int) -> tuple[int, str]:
        """Devuelve (id, prompt). Los ids no se repiten dentro de una tanda."""
        i = self._siguiente
        self._siguiente += 1
        nonce = self._rng.randrange(10**9)
        return i, f"Solicitud {i:06d}-{nonce:09d}. {cuerpo(n_palabras)}\nResume el texto anterior:"


def dentro_de_tolerancia(reales: int, pedidos: int, tol: float = TOLERANCIA_ENTRADA) -> bool:
    return abs(reales - pedidos) <= tol * pedidos


@dataclass
class Calibracion:
    """Resultado de calibrar un punto: cuántas palabras dan ~`objetivo` tokens con este modelo."""

    objetivo: int
    palabras: int
    tokens: int                                   # prompt_eval_count del último sondeo
    sondeos: list[dict] = field(default_factory=list)  # [{"palabras", "tokens"}], en orden

    @property
    def ok(self) -> bool:
        return dentro_de_tolerancia(self.tokens, self.objetivo)

    def a_dict(self) -> dict:
        return {"objetivo": self.objetivo, "palabras": self.palabras, "tokens": self.tokens,
                "ok": self.ok, "tolerancia": TOLERANCIA_ENTRADA, "sondeos": self.sondeos}


async def calibrar(
    objetivo: int,
    medir_tokens: Callable[[int], Awaitable[int]],
    max_sondeos: int = MAX_SONDEOS,
) -> Calibracion:
    """Busca el número de palabras cuyo prompt da `objetivo` tokens (±2 %).

    `medir_tokens(palabras)` lanza un sondeo y devuelve `prompt_eval_count`. El modelo es lineal
    (tokens ≈ a + b·palabras): se estima con los dos últimos sondeos y se corrige; para en cuanto el
    error es ≤ 1 token y además cabe en la tolerancia de G7 (con objetivos pequeños, 1 token ya es
    más del 2 %), o cuando se acaban los sondeos. No inventa nada: si no converge, `ok` es False y la
    guardia G7 falla con los sondeos a la vista.
    """
    sondeos: list[dict] = []

    async def sondear(p: int) -> int:
        t = await medir_tokens(p)
        sondeos.append({"palabras": p, "tokens": t})
        return t

    p = max(1, round(objetivo * 0.75))            # primera aproximación: ~1,33 tokens por palabra
    t = await sondear(p)
    mejor = (abs(t - objetivo), p, t)
    pendiente = 1.33
    umbral = min(1, int(TOLERANCIA_ENTRADA * objetivo))
    while len(sondeos) < max_sondeos and abs(t - objetivo) > umbral:
        if len(sondeos) >= 2:
            (p0, t0), (p1, t1) = ((s["palabras"], s["tokens"]) for s in sondeos[-2:])
            if p1 != p0 and (t1 - t0) / (p1 - p0) > 0:
                pendiente = (t1 - t0) / (p1 - p0)
        nuevo = max(1, p + round((objetivo - t) / pendiente))
        if nuevo == p:                            # el paso redondea a cero: probar el vecino
            nuevo = p + (1 if t < objetivo else -1)
            if nuevo < 1:
                break
        p = nuevo
        t = await sondear(p)
        if abs(t - objetivo) < mejor[0]:
            mejor = (abs(t - objetivo), p, t)
    _, p, t = mejor
    return Calibracion(objetivo=objetivo, palabras=p, tokens=t, sondeos=sondeos)
