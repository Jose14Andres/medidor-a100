"""Escritura de los JSON de resultados con esquema versionado (DESARROLLO.md §5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ESQUEMA = "medidor-a100/v1"


def escribir_json(ruta: Path, datos: dict[str, Any]) -> None:
    """Escribe `datos` con la clave `esquema` al principio.

    Nunca sobrescribe: si el fichero ya existe, lanza FileExistsError (§6, "no sobrescribe datos").
    """
    if "esquema" in datos and datos["esquema"] != ESQUEMA:
        raise ValueError(f"esquema {datos['esquema']!r} distinto de {ESQUEMA!r}")
    contenido = {"esquema": ESQUEMA, **{k: v for k, v in datos.items() if k != "esquema"}}
    with open(ruta, "x", encoding="utf-8") as f:
        json.dump(contenido, f, ensure_ascii=False, indent=2)
        f.write("\n")
