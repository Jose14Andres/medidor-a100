"""Interfaz de línea de órdenes (DESARROLLO.md §7).

Códigos de salida: 0 bien, 1 fallo de guardia, 2 error de configuración, 3 tanda con errores parciales.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

from medidor import __version__, config
from medidor.esquema import ESQUEMA, escribir_json
from medidor.tanda import (DEPENDENCIAS, SALIDA_CONFIG, SALIDA_GUARDIA, SALIDA_OK, SALIDA_PARCIAL,
                           ejecutar, resumen_desde_datos, version_paquete)

__all__ = ["SALIDA_OK", "SALIDA_GUARDIA", "SALIDA_CONFIG", "SALIDA_PARCIAL", "main", "construir_parser"]


def _orden_entorno(args: argparse.Namespace) -> int:
    """Versiones y validez de la configuración. No toca la red."""
    info: dict = {
        "medidor": __version__,
        "esquema": ESQUEMA,
        "python": platform.python_version(),
        "python_ejecutable": sys.executable,
        "plataforma": platform.platform(),
        "dependencias": {p: version_paquete(p) for p in DEPENDENCIAS},
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


def _cargar_o_salir(ruta: Path) -> config.Config | None:
    try:
        return config.cargar(ruta)
    except config.ErrorConfig as e:
        print(f"config NO VÁLIDA: {e}", file=sys.stderr)
        return None


def _orden_comprobar(args: argparse.Namespace) -> int:
    """Guardias G1–G3, G5–G8 con sondeos, sin medir ni escribir resultados."""
    cfg = _cargar_o_salir(args.config)
    if cfg is None:
        return SALIDA_CONFIG
    inf = asyncio.run(ejecutar(cfg, args.config, medir_tanda=False))
    if args.json:
        print(json.dumps({"codigo": inf.codigo, "guardias": [g.a_dict() for g in inf.guardias]},
                         ensure_ascii=False, indent=2))
    return inf.codigo


def _orden_medir(args: argparse.Namespace) -> int:
    """Guardias + tanda completa + resumen, en una carpeta nueva de resultados."""
    cfg = _cargar_o_salir(args.config)
    if cfg is None:
        return SALIDA_CONFIG
    inf = asyncio.run(ejecutar(cfg, args.config, medir_tanda=True))
    if inf.dir is not None:
        print(f"resultados: {inf.dir}")
    return inf.codigo


def _orden_resumir(args: argparse.Namespace) -> int:
    """Regenera el resumen desde datos.json. No sobrescribe: si ya hay resumen.json, crea otro."""
    d: Path = args.dir
    try:
        datos = json.loads((d / "datos.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"no se puede leer {d / 'datos.json'}: {e}", file=sys.stderr)
        return SALIDA_CONFIG
    if datos.get("esquema") != ESQUEMA:
        print(f"{d / 'datos.json'}: esquema {datos.get('esquema')!r}, se esperaba {ESQUEMA!r}", file=sys.stderr)
        return SALIDA_CONFIG
    destino = d / "resumen.json"
    if destino.exists():
        destino = d / f"resumen-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    escribir_json(destino, resumen_desde_datos(datos))
    print(f"resumen escrito en {destino}")
    return SALIDA_OK


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medidor", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"medidor {__version__}")
    sub = parser.add_subparsers(dest="orden", required=True, metavar="ORDEN")

    p = sub.add_parser("entorno", help="versiones, Python y validez de la configuración (sin red)")
    p.add_argument("-c", "--config", type=Path, help="TOML de experimento a validar")
    p.add_argument("--json", action="store_true", help="salida en JSON")
    p.set_defaults(func=_orden_entorno)

    p = sub.add_parser("comprobar", help="guardias G1–G3, G5–G8 sin medir")
    p.add_argument("-c", "--config", type=Path, required=True, help="TOML de experimento")
    p.add_argument("--json", action="store_true", help="además, resultado de las guardias en JSON")
    p.set_defaults(func=_orden_comprobar)

    p = sub.add_parser("medir", help="guardias + tanda completa + resumen")
    p.add_argument("-c", "--config", type=Path, required=True, help="TOML de experimento")
    p.set_defaults(func=_orden_medir)

    p = sub.add_parser("resumir", help="regenera el resumen desde datos.json (sin sobrescribir)")
    p.add_argument("dir", type=Path, help="carpeta de una tanda (resultados/<nombre>-<fecha>)")
    p.set_defaults(func=_orden_resumir)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    return args.func(args)
