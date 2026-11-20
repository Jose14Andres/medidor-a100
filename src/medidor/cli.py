"""Interfaz de línea de órdenes (DESARROLLO.md §7).

Códigos de salida: 0 bien, 1 fallo de guardia, 2 error de configuración, 3 tanda con errores parciales.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

from medidor import __version__, config
from medidor.esquema import ESQUEMA

SALIDA_OK = 0
SALIDA_GUARDIA = 1
SALIDA_CONFIG = 2
SALIDA_PARCIAL = 3

DEPENDENCIAS = ("httpx", "matplotlib", "numpy")


def _version(paquete: str) -> str | None:
    try:
        return metadata.version(paquete)
    except metadata.PackageNotFoundError:
        return None


def _orden_entorno(args: argparse.Namespace) -> int:
    """Versiones y validez de la configuración. No toca la red."""
    info: dict = {
        "medidor": __version__,
        "esquema": ESQUEMA,
        "python": platform.python_version(),
        "python_ejecutable": sys.executable,
        "plataforma": platform.platform(),
        "dependencias": {p: _version(p) for p in DEPENDENCIAS},
        "config": None,
    }
    codigo = SALIDA_OK
    if args.config is not None:
        try:
            cfg = config.cargar(args.config)
            info["config"] = {
                "ruta": str(args.config),
                "valida": True,
                "experimento": cfg.experimento.nombre,
                "backend": cfg.servidor.backend,
                "modelo": cfg.servidor.modelo,
            }
        except config.ErrorConfig as e:
            info["config"] = {"ruta": str(args.config), "valida": False, "error": str(e)}
            codigo = SALIDA_CONFIG

    faltan = [p for p, v in info["dependencias"].items() if v is None]

    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print(f"medidor      {info['medidor']}  (esquema {info['esquema']})")
        print(f"python       {info['python']}  {info['python_ejecutable']}")
        print(f"plataforma   {info['plataforma']}")
        for p, v in info["dependencias"].items():
            print(f"{p:<12} {v or 'NO INSTALADO'}")
        c = info["config"]
        if c is not None:
            if c["valida"]:
                print(f"config       válida: {c['ruta']} ({c['experimento']}, {c['backend']}, {c['modelo']})")
            else:
                print(f"config       NO VÁLIDA: {c['error']}")
        if faltan:
            print(f"aviso: faltan dependencias ({', '.join(faltan)}); instala con `pip install -e .`")
    return codigo


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medidor", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"medidor {__version__}")
    sub = parser.add_subparsers(dest="orden", required=True, metavar="ORDEN")

    p = sub.add_parser("entorno", help="versiones, Python y validez de la configuración (sin red)")
    p.add_argument("-c", "--config", type=Path, help="TOML de experimento a validar")
    p.add_argument("--json", action="store_true", help="salida en JSON")
    p.set_defaults(func=_orden_entorno)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    return args.func(args)
