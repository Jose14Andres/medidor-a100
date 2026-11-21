"""Agregados por punto de medida a partir de `datos.json` (DESARROLLO.md §2 y §5).

Solo números medidos: las solicitudes de calentamiento y las que tienen error no entran en los
estadísticos, pero se cuentan. Un punto con errores muestra su `n` real; no se rellena nada.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np

# Métricas por solicitud que se resumen; (nombre en el resumen, función que la extrae).
METRICAS: dict[str, Any] = {
    "ttft_s": lambda s: s.get("ttft_s"),
    "e2e_s": lambda s: s.get("e2e_s"),
    "tpot_s": lambda s: s.get("tpot_s"),
    "tokens_entrada": lambda s: s.get("tokens_entrada"),
    "tokens_salida": lambda s: s.get("tokens_salida"),
    "prefill_tok_s_cliente": lambda s: s.get("prefill_tok_s_cliente"),
    "decode_tok_s_cliente": lambda s: s.get("decode_tok_s_cliente"),
    "servidor_prefill_tok_s": lambda s: (s.get("servidor") or {}).get("prefill_tok_s"),
    "servidor_decode_tok_s": lambda s: (s.get("servidor") or {}).get("decode_tok_s"),
    "servidor_prompt_eval_duration_s": lambda s: (s.get("servidor") or {}).get("prompt_eval_duration_s"),
    "servidor_eval_duration_s": lambda s: (s.get("servidor") or {}).get("eval_duration_s"),
    "servidor_load_duration_s": lambda s: (s.get("servidor") or {}).get("load_duration_s"),
}


def estadisticos(valores: Iterable[float | None]) -> dict | None:
    """n, media, desviación típica (muestral), mín, p50, p90, p95, p99, máx. None si no hay datos."""
    xs = np.array([v for v in valores if v is not None], dtype=float)
    if xs.size == 0:
        return None
    p50, p90, p95, p99 = np.percentile(xs, [50, 90, 95, 99])
    return {
        "n": int(xs.size), "media": float(xs.mean()),
        "desv": float(xs.std(ddof=1)) if xs.size > 1 else 0.0,
        "min": float(xs.min()), "p50": float(p50), "p90": float(p90), "p95": float(p95),
        "p99": float(p99), "max": float(xs.max()),
    }


def _clave(punto: dict) -> tuple[int, int, int]:
    return (punto["entrada"], punto["salida"], punto["concurrencia"])


def resumir(solicitudes: Sequence[dict]) -> list[dict]:
    """Un agregado por punto (entrada, salida, concurrencia), en el orden en que aparecen."""
    grupos: dict[tuple, list[dict]] = {}
    for s in solicitudes:
        grupos.setdefault(_clave(s["punto"]), []).append(s)

    puntos = []
    for (entrada, salida, conc), todas in grupos.items():
        medidas = [s for s in todas if not s.get("calentamiento")]
        ok = [s for s in medidas if s.get("error") is None]
        itl = [x for s in ok for x in s.get("itl_s") or []]
        puntos.append({
            "punto": {"entrada": entrada, "salida": salida, "concurrencia": conc},
            "n_solicitudes": len(medidas),
            "n_ok": len(ok),
            "n_errores": len(medidas) - len(ok),
            "n_calentamiento": len(todas) - len(medidas),
            "errores": [s["error"] for s in medidas if s.get("error") is not None],
            "finish_reason": dict(Counter(s.get("finish_reason") for s in ok)),
            # Ollama no tiene ignore_eos: una salida corta cambia el punto; se cuenta, no se oculta.
            "n_salida_corta": sum(1 for s in ok if (s.get("tokens_salida") or 0) < salida),
            "n_entrada_fuera_tolerancia": sum(1 for s in ok if s.get("entrada_fuera_tolerancia")),
            **{m: estadisticos(f(s) for s in ok) for m, f in METRICAS.items()},
            "itl_s": estadisticos(itl),
        })
    return puntos
