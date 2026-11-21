"""Nodo, job de Slurm y GPU vistos desde el cliente (DESARROLLO.md §3.2 y §11).

Fase 1: solo lo necesario para las guardias y `entorno.json` en modo nodo (cliente en el propio nodo
GPU). El túnel SSH y el RTT del modo B llegan con `medidor nodo`.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess

TIMEOUT_NVIDIA_SMI_S = 20.0   # el mismo que medir_ollama.py; con loadavg alto nvidia-smi tarda


class ErrorConexion(Exception):
    pass


def _nvidia_smi(*args: str) -> str:
    if shutil.which("nvidia-smi") is None:
        raise ErrorConexion("nvidia-smi no está en el PATH: este cliente no corre en un nodo con GPU")
    try:
        r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                           timeout=TIMEOUT_NVIDIA_SMI_S)
    except subprocess.TimeoutExpired as e:
        raise ErrorConexion(f"nvidia-smi no respondió en {TIMEOUT_NVIDIA_SMI_S:.0f} s") from e
    if r.returncode != 0:
        raise ErrorConexion(f"nvidia-smi salió con {r.returncode}: {(r.stderr or r.stdout).strip()[:300]}")
    return r.stdout


def _entero(v: str) -> int | None:
    v = v.strip()
    return int(v) if v.isdigit() else None   # "[N/A]" en trozos MIG


def memoria_gpu() -> list[dict]:
    """Una entrada por GPU visible: índice, nombre, memoria usada/total (MiB), driver."""
    salida = _nvidia_smi("--query-gpu=index,name,memory.used,memory.total,driver_version",
                         "--format=csv,noheader,nounits")
    gpus = []
    for linea in salida.strip().splitlines():
        partes = [p.strip() for p in linea.split(",")]
        if len(partes) != 5:
            continue
        gpus.append({"indice": _entero(partes[0]), "nombre": partes[1], "usado_mib": _entero(partes[2]),
                     "total_mib": _entero(partes[3]), "driver": partes[4]})
    if not gpus:
        raise ErrorConexion(f"nvidia-smi no devolvió ninguna GPU: {salida.strip()[:300]!r}")
    return gpus


def gpus_L() -> list[str]:
    """`nvidia-smi -L`: la GPU real (incluye los trozos MIG, que `nvidia-smi` a secas no muestra)."""
    return [l.strip() for l in _nvidia_smi("-L").splitlines() if l.strip()]


def loadavg() -> list[float] | None:
    try:
        with open("/proc/loadavg", encoding="ascii") as f:
            return [float(x) for x in f.read().split()[:3]]
    except OSError:
        return None


def info_nodo() -> dict:
    return {
        "nodo": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_particion": os.environ.get("SLURM_JOB_PARTITION"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def instantanea_gpu() -> dict:
    """GPU para `entorno.json`; si no se puede leer, guarda el error en lugar de fallar."""
    d: dict = {}
    try:
        d["nvidia_smi_L"] = gpus_L()
        d["memoria"] = memoria_gpu()
    except ErrorConexion as e:
        d["error"] = str(e)
    return d
