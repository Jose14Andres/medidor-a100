"""Guardias (preflight) obligatorias antes de medir (DESARROLLO.md §6).

Si una guardia falla, el programa para y explica por qué: no reintenta ni improvisa otra cosa. Nunca
mata procesos ni sube tiempos de espera. Fase 1 implementa G1–G3 y G5–G8 para Ollama (G4, la GPU
esperada, llega con `medidor nodo`).

Cada guardia devuelve un `ResultadoGuardia`; el orden lo fija `tanda.py`, porque G3 y G2 solo
tienen sentido con el modelo ya cargado y G7 necesita sondeos.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from medidor import conexion
from medidor.prompts import TOLERANCIA_ENTRADA, Calibracion

# G8: los resultados van a /home. /scratch/$USER/<jobid> lo borra el epílogo de Slurm (§11).
RAICES_PERMITIDAS: tuple[Path, ...] = (Path("/home"),)


@dataclass
class ResultadoGuardia:
    id: str
    ok: bool
    mensaje: str
    datos: dict = field(default_factory=dict)

    def a_dict(self) -> dict:
        return asdict(self)


# --- G1, G5, G6: servidor y configuración ------------------------------------------------------

async def g1_salud(backend) -> ResultadoGuardia:
    e = await backend.salud()
    if not e.ok:
        return ResultadoGuardia("G1", False, f"servidor no listo: {e.detalle}")
    return ResultadoGuardia("G1", True, e.detalle, {"version": e.version})


async def g5_modelo(backend, modelo: str) -> ResultadoGuardia:
    disponibles = await backend.modelos()
    if modelo not in disponibles:
        return ResultadoGuardia("G5", False, f"el modelo {modelo!r} no está en el servidor; hay: "
                                f"{', '.join(disponibles) or '(ninguno)'}", {"disponibles": disponibles})
    return ResultadoGuardia("G5", True, f"modelo {modelo!r} disponible", {"disponibles": disponibles})


def g6_contexto(contexto: int, entradas: Sequence[int], salidas: Sequence[int],
                nombre: str = "num_ctx") -> ResultadoGuardia:
    # Los tokens reales pueden pasar de los pedidos hasta la tolerancia de G7.
    necesario = int(max(entradas) * (1 + TOLERANCIA_ENTRADA)) + max(salidas)
    datos = {nombre: contexto, "necesario": necesario}
    if contexto < necesario:
        return ResultadoGuardia("G6", False, f"{nombre} = {contexto} < entrada máx (+{TOLERANCIA_ENTRADA:.0%}) "
                                f"+ salida máx = {necesario}: Ollama truncaría el prompt en silencio", datos)
    return ResultadoGuardia("G6", True, f"{nombre} = {contexto} ≥ {necesario}", datos)


# --- G8: directorio de salida ------------------------------------------------------------------

def g8_directorio(ruta: Path) -> ResultadoGuardia:
    real = ruta.expanduser().resolve()
    raices = [r.expanduser().resolve() for r in RAICES_PERMITIDAS]   # en Windows /home → C:\home
    if not any(real == r or r in real.parents for r in raices):
        return ResultadoGuardia("G8", False, f"el directorio de salida {real} no está en "
                                f"{', '.join(map(str, RAICES_PERMITIDAS))} (/scratch se borra al acabar el job)",
                                {"ruta": str(real)})
    return ResultadoGuardia("G8", True, f"salida en {real}", {"ruta": str(real)})


# --- G3: Ollama en GPU, no en CPU --------------------------------------------------------------

_RE_LIBRERIA = re.compile(r'msg="inference compute".*?\blibrary=(\S+)')
_RE_OFFLOAD = re.compile(r'offloaded (\d+)/(\d+) layers to GPU')


def expandir_log(plantilla: str) -> Path:
    """`~` y `{job}` (SLURM_JOB_ID). Lanza KeyError si hace falta {job} y no hay job."""
    if "{job}" in plantilla:
        job = os.environ.get("SLURM_JOB_ID")
        if not job:
            raise KeyError("SLURM_JOB_ID")
        plantilla = plantilla.replace("{job}", job)
    return Path(plantilla).expanduser()


def analizar_log_ollama(texto: str) -> dict:
    """Lo que dice el log de `ollama serve` desde su último arranque ("Listening on")."""
    lineas = texto.splitlines()
    inicio = max((i for i, l in enumerate(lineas) if "Listening on" in l), default=0)
    lineas = lineas[inicio:]
    librerias = [m.group(1).strip('"') for l in lineas if (m := _RE_LIBRERIA.search(l))]
    offloads = [(int(m.group(1)), int(m.group(2))) for l in lineas if (m := _RE_OFFLOAD.search(l))]
    return {
        "librerias": librerias,
        "fallos_descubrimiento": sum("failure during GPU discovery" in l for l in lineas),
        "offload": list(offloads[-1]) if offloads else None,
    }


async def g3_ollama_gpu(backend, modelo: str, plantilla_log: str) -> ResultadoGuardia:
    """Log con `inference compute` CUDA y todas las capas en GPU, y `/api/ps` con size_vram > 0.

    El caso que evita: el 17-nov (job 36509) el descubrimiento de GPU superó sus 30 s y Ollama cayó a
    CPU en silencio (`library=cpu`, `offloaded 0/49`, `size_vram 0`).
    """
    try:
        ruta = expandir_log(plantilla_log)
    except KeyError:
        return ResultadoGuardia("G3", False, f"el log {plantilla_log!r} usa {{job}} pero no hay SLURM_JOB_ID")
    datos: dict = {"log": str(ruta)}
    try:
        texto = ruta.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ResultadoGuardia("G3", False, f"no se puede leer el log de Ollama {ruta}: {e}", datos)
    log = analizar_log_ollama(texto)
    datos.update(log)

    problemas = []
    if not any(l.upper() == "CUDA" for l in log["librerias"]):
        problemas.append(f"el log no tiene 'inference compute' con CUDA (librerías: {log['librerias'] or 'ninguna'}"
                         f"; fallos de descubrimiento de GPU: {log['fallos_descubrimiento']})")
    if log["offload"] is None:
        problemas.append("el log no dice cuántas capas se cargaron en GPU ('offloaded N/M layers')")
    elif log["offload"][0] != log["offload"][1]:
        problemas.append(f"solo {log['offload'][0]}/{log['offload'][1]} capas en GPU")

    cargados = await backend.modelos_cargados()
    m = next((x for x in cargados if (x.get("name") or x.get("model")) == modelo), None)
    if m is None:
        problemas.append(f"/api/ps no muestra {modelo!r} cargado")
    else:
        datos["size"], datos["size_vram"] = m.get("size"), m.get("size_vram")
        if not m.get("size_vram"):
            problemas.append("/api/ps da size_vram = 0: el modelo está en CPU")
        elif m.get("size") and m["size_vram"] < m["size"]:
            problemas.append(f"/api/ps: solo {m['size_vram'] / m['size']:.0%} del modelo en VRAM")

    if problemas:
        return ResultadoGuardia("G3", False, "Ollama no está entero en GPU: " + "; ".join(problemas), datos)
    return ResultadoGuardia("G3", True, f"Ollama en GPU (CUDA, {log['offload'][0]}/{log['offload'][1]} capas, "
                            f"size_vram {datos['size_vram'] / 2**30:.1f} GiB)", datos)


# --- G2: memoria GPU en uso --------------------------------------------------------------------

def g2_memoria_gpu(consulta: Callable[[], list[dict]] | None = None) -> ResultadoGuardia:
    try:
        gpus = (consulta or conexion.memoria_gpu)()
    except conexion.ErrorConexion as e:
        return ResultadoGuardia("G2", False, f"no se puede comprobar la GPU: {e}")
    usada = [g["usado_mib"] for g in gpus if g.get("usado_mib") is not None]
    if not usada:
        return ResultadoGuardia("G2", False, "nvidia-smi no informa de la memoria usada", {"gpus": gpus})
    if max(usada) <= 0:
        return ResultadoGuardia("G2", False, "el servidor responde pero no está en GPU (memoria usada 0 MiB)",
                                {"gpus": gpus})
    return ResultadoGuardia("G2", True, f"memoria GPU en uso: {max(usada)} MiB", {"gpus": gpus})


# --- G7: tokens de entrada reales ≈ pedidos ----------------------------------------------------

def g7_entrada(calibraciones: Sequence[Calibracion]) -> ResultadoGuardia:
    datos = {"puntos": [c.a_dict() for c in calibraciones]}
    malas = [c for c in calibraciones if not c.ok]
    if malas:
        detalle = ", ".join(f"pedidos {c.objetivo} → reales {c.tokens} (con {c.palabras} palabras)" for c in malas)
        return ResultadoGuardia("G7", False, f"los tokens de entrada no llegan a ±{TOLERANCIA_ENTRADA:.0%} de los "
                                f"pedidos: {detalle}. ¿Truncado por num_ctx o tokenizador inesperado?", datos)
    detalle = ", ".join(f"{c.objetivo}→{c.tokens}" for c in calibraciones)
    return ResultadoGuardia("G7", True, f"entrada calibrada (pedidos→reales): {detalle}", datos)
