"""Comparación de una tanda con la referencia de `medir_ollama.py` (FASE-1.md, tarea 1.1).

Criterio de la fase 1: prefill y decode tok/s del servidor (p50 del medidor frente a la mediana de la
referencia) dentro de ±5 %, punto a punto. Los puntos se emparejan por tokens reales de entrada, uno
a uno y por cercanía; si los tokens difieren más de un 2 % se avisa, pero no cambia el veredicto.
TTFT y TPOT del cliente se informan sin entrar en el criterio.

Solo números medidos: un punto de la referencia sin pareja, o una métrica sin valor, es NO PASA.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from medidor.esquema import ESQUEMA, escribir_json

TOLERANCIA = 0.05            # ±5 % en tok/s del servidor
TOLERANCIA_TOKENS = 0.02     # ±2 % en tokens de entrada: solo aviso
_EPS = 1e-12                 # para que el borde exacto (5,0 %) no dependa del redondeo

# (nombre en la comparación, clave en el medidor [p50], clave en la referencia [mediana])
CRITERIO = [
    ("servidor_prefill_tok_s", "servidor_prefill_tok_s", "servidor_prefill_tok_s"),
    ("servidor_decode_tok_s", "servidor_decode_tok_s", "servidor_decode_tok_s"),
]
INFORMATIVO = [
    ("ttft_cliente_s", "ttft_s", "ttft_s"),
    ("tpot_cliente_s", "tpot_s", "tpot_cliente_s"),
]


class ErrorComparar(Exception):
    """Entrada ilegible o con un formato que no es el esperado."""


def _leer(ruta: Path) -> dict:
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ErrorComparar(f"no se puede leer {ruta}: {e}") from e


def _valor(d: dict, metrica: str, estadistico: str) -> float | None:
    return (d.get(metrica) or {}).get(estadistico)


def leer_medidor(ruta: Path) -> list[dict]:
    """Puntos del `resumen.json` del medidor, con los campos que usa la comparación."""
    datos = _leer(ruta)
    if datos.get("esquema") != ESQUEMA:
        raise ErrorComparar(f"{ruta}: esquema {datos.get('esquema')!r}, se esperaba {ESQUEMA!r}")
    try:
        puntos = []
        for p in datos["puntos"]:
            tokens = _valor(p, "tokens_entrada", "media")
            puntos.append({
                "punto": p["punto"],
                # Sin solicitudes correctas no hay tokens medidos: se empareja por los pedidos.
                "tokens": tokens if tokens is not None else p["punto"]["entrada"],
                "tokens_medidos": tokens is not None,
                "valores": {n: _valor(p, k, "p50") for n, k, _ in CRITERIO + INFORMATIVO},
                "diagnostico": {k: p.get(k) for k in
                                ("n_ok", "n_errores", "n_salida_corta", "n_entrada_fuera_tolerancia")},
            })
    except (KeyError, TypeError) as e:
        raise ErrorComparar(f"{ruta}: no tiene el formato de resumen del medidor ({e!r})") from e
    return puntos


def leer_referencia(ruta: Path) -> tuple[dict, list[dict]]:
    """`meta` y puntos del `resumen.json` de `medir_ollama.py` (no lleva clave `esquema`)."""
    datos = _leer(ruta)
    try:
        puntos = [{
            "entrada_palabras": p["entrada_palabras"],
            "tokens": p["prompt_tokens"]["media"],
            "valores": {n: _valor(p, k, "mediana") for n, _, k in CRITERIO + INFORMATIVO},
        } for p in datos["puntos"]]
    except (KeyError, TypeError) as e:
        raise ErrorComparar(f"{ruta}: no tiene el formato de resumen de medir_ollama.py ({e!r})") from e
    return datos.get("meta") or {}, puntos


def emparejar(referencia: list[dict], nuevos: list[dict]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Parejas (i_ref, i_nuevo) uno a uno, de la más cercana en tokens a la más lejana.

    Devuelve también los índices sin pareja de cada lado.
    """
    candidatos = sorted((abs(n["tokens"] - r["tokens"]), i, j)
                        for i, r in enumerate(referencia) for j, n in enumerate(nuevos))
    usados_r: set[int] = set()
    usados_n: set[int] = set()
    parejas = []
    for _, i, j in candidatos:
        if i not in usados_r and j not in usados_n:
            parejas.append((i, j))
            usados_r.add(i)
            usados_n.add(j)
    parejas.sort()
    return (parejas, [i for i in range(len(referencia)) if i not in usados_r],
            [j for j in range(len(nuevos)) if j not in usados_n])


def _dif_rel(ref: float | None, nuevo: float | None) -> float | None:
    if ref is None or nuevo is None or ref == 0:
        return None
    return (nuevo - ref) / ref


def _evaluar(ref: float | None, nuevo: float | None) -> dict:
    dif = _dif_rel(ref, nuevo)
    if dif is None:
        motivo = "sin valor en la tanda" if nuevo is None else "sin valor en la referencia"
        return {"referencia": ref, "nuevo": nuevo, "dif_rel": None, "pasa": False, "motivo": motivo}
    return {"referencia": ref, "nuevo": nuevo, "dif_rel": dif, "pasa": abs(dif) <= TOLERANCIA + _EPS,
            "motivo": None}


def comparar(referencia: list[dict], nuevos: list[dict]) -> dict:
    """Resultado de la comparación (sin rutas ni marcas de tiempo)."""
    parejas, sin_r, sin_n = emparejar(referencia, nuevos)
    avisos: list[str] = []
    resultado = []
    for i, j in parejas:
        r, n = referencia[i], nuevos[j]
        dif_tokens = _dif_rel(r["tokens"], n["tokens"])
        aviso_tokens = dif_tokens is None or abs(dif_tokens) > TOLERANCIA_TOKENS
        if aviso_tokens:
            avisos.append(f"{r['entrada_palabras']} palabras: tokens de entrada {n['tokens']} frente a "
                          f"{r['tokens']} de la referencia (fuera de ±{TOLERANCIA_TOKENS:.0%})")
        if not n["tokens_medidos"]:
            avisos.append(f"punto {n['punto']['entrada']}: sin tokens medidos, emparejado por los pedidos")
        criterio = {nombre: _evaluar(r["valores"][nombre], n["valores"][nombre]) for nombre, _, _ in CRITERIO}
        resultado.append({
            "referencia": {"entrada_palabras": r["entrada_palabras"], "tokens": r["tokens"]},
            "nuevo": {"punto": n["punto"], "tokens": n["tokens"], "tokens_medidos": n["tokens_medidos"],
                      "diagnostico": n["diagnostico"]},
            "dif_tokens_rel": dif_tokens,
            "aviso_tokens": aviso_tokens,
            "criterio": criterio,
            "informativo": {nombre: {"referencia": r["valores"][nombre], "nuevo": n["valores"][nombre],
                                     "dif_rel": _dif_rel(r["valores"][nombre], n["valores"][nombre])}
                            for nombre, _, _ in INFORMATIVO},
            "pasa": all(c["pasa"] for c in criterio.values()),
        })
    for i in sin_r:
        avisos.append(f"{referencia[i]['entrada_palabras']} palabras ({referencia[i]['tokens']} tokens) "
                      "de la referencia no tiene pareja en la tanda")
    return {
        "criterio": {"metricas": [n for n, _, _ in CRITERIO], "tolerancia_rel": TOLERANCIA,
                     "tolerancia_tokens_rel": TOLERANCIA_TOKENS,
                     "estadistico_nuevo": "p50", "estadistico_referencia": "mediana"},
        "pasa": bool(resultado) and not sin_r and all(p["pasa"] for p in resultado),
        "parejas": resultado,
        "referencia_sin_pareja": [referencia[i] for i in sin_r],
        "nuevo_sin_pareja": [{"punto": nuevos[j]["punto"], "tokens": nuevos[j]["tokens"]} for j in sin_n],
        "avisos": avisos,
    }


def destino_libre(d: Path) -> Path:
    """`comparacion.json`, o `comparacion-<fecha>[-k].json` si ya existe (nunca sobrescribe)."""
    destino = d / "comparacion.json"
    if not destino.exists():
        return destino
    base = f"comparacion-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    destino, k = d / f"{base}.json", 2
    while destino.exists():
        destino, k = d / f"{base}-{k}.json", k + 1
    return destino


def ejecutar(dir_tanda: Path, ruta_referencia: Path) -> tuple[dict, Path]:
    """Lee, compara y escribe la comparación en la carpeta de la tanda."""
    ruta_resumen = dir_tanda / "resumen.json"
    nuevos = leer_medidor(ruta_resumen)
    meta, referencia = leer_referencia(ruta_referencia)
    res = {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "tanda": str(dir_tanda), "resumen_tanda": str(ruta_resumen), "referencia": str(ruta_referencia),
        "referencia_meta": meta,
        **comparar(referencia, nuevos),
    }
    destino = destino_libre(dir_tanda)
    escribir_json(destino, res)
    return res, destino


def _num(x: float | None, fmt: str) -> str:
    return "—" if x is None else format(x, fmt)


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:+.2%}"


def tabla(res: dict) -> str:
    """Tabla legible: una fila por métrica del criterio y otra por cada métrica informativa."""
    filas = [f"{'punto':<22} {'métrica':<24} {'referencia':>11} {'nuevo':>11} {'dif':>8}  veredicto"]
    for p in res["parejas"]:
        etiqueta = f"{p['referencia']['entrada_palabras']} pal / {_num(p['nuevo']['tokens'], '.0f')} tok"
        for nombre, c in p["criterio"].items():
            veredicto = "PASA" if c["pasa"] else f"NO PASA{' (' + c['motivo'] + ')' if c['motivo'] else ''}"
            filas.append(f"{etiqueta:<22} {nombre:<24} {_num(c['referencia'], '11.1f')} "
                         f"{_num(c['nuevo'], '11.1f')} {_pct(c['dif_rel']):>8}  {veredicto}")
        for nombre, c in p["informativo"].items():
            filas.append(f"{etiqueta:<22} {nombre:<24} {_num(c['referencia'], '11.4f')} "
                         f"{_num(c['nuevo'], '11.4f')} {_pct(c['dif_rel']):>8}  (informativo)")
    for aviso in res["avisos"]:
        filas.append(f"aviso: {aviso}")
    for n in res["nuevo_sin_pareja"]:
        filas.append(f"info: el punto {n['punto']['entrada']} de la tanda no está en la referencia")
    filas.append(f"resultado: {'PASA' if res['pasa'] else 'NO PASA'} (±{res['criterio']['tolerancia_rel']:.0%})")
    return "\n".join(filas)
