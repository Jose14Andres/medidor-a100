"""Carga y validación estricta de la configuración TOML de un experimento (DESARROLLO.md §4).

Cualquier clave desconocida, tipo incorrecto o combinación inválida es un error: una errata en la
configuración invalida la tanda entera, así que se detecta antes de medir.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

BACKENDS = ("vllm", "ollama")
MODOS_CARGA = ("cerrado", "poisson")


class ErrorConfig(Exception):
    """Configuración inválida. El mensaje indica la clave (`seccion.clave`) y el motivo."""


@dataclass(frozen=True)
class Experimento:
    nombre: str
    descripcion: str


@dataclass(frozen=True)
class OpcionesOllama:
    num_ctx: int
    keep_alive: str


@dataclass(frozen=True)
class Servidor:
    backend: str
    url: str
    modelo: str
    gpu_esperada: str | None
    opciones_ollama: OpcionesOllama | None


@dataclass(frozen=True)
class Carga:
    entradas: tuple[int, ...]
    salidas: tuple[int, ...]
    concurrencias: tuple[int, ...]
    repeticiones: int
    calentamiento: int
    modo: str
    tasa_rps: float | None
    semilla: int
    temperatura: float


@dataclass(frozen=True)
class Telemetria:
    activar: bool
    intervalo_ms: int


@dataclass(frozen=True)
class Salida:
    dir: str


@dataclass(frozen=True)
class Config:
    experimento: Experimento
    servidor: Servidor
    carga: Carga
    telemetria: Telemetria
    salida: Salida


_OBLIGATORIA = object()


def _tipo_ok(valor: Any, tipo: type) -> bool:
    # bool es subclase de int en Python: `repeticiones = true` no debe colar como 1.
    if tipo is int:
        return isinstance(valor, int) and not isinstance(valor, bool)
    if tipo is float:
        return isinstance(valor, (int, float)) and not isinstance(valor, bool)
    return isinstance(valor, tipo)


def _leer_tabla(
    tabla: Any, ruta: str, campos: dict[str, tuple[type, Any]], subtablas: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Valida una tabla TOML contra `campos` ({clave: (tipo, defecto | _OBLIGATORIA)})."""
    if not isinstance(tabla, dict):
        raise ErrorConfig(f"[{ruta}] debe ser una tabla")
    desconocidas = sorted(set(tabla) - set(campos) - set(subtablas))
    if desconocidas:
        raise ErrorConfig(f"[{ruta}] claves desconocidas: {', '.join(desconocidas)}")
    salida: dict[str, Any] = {}
    for clave, (tipo, defecto) in campos.items():
        if clave not in tabla:
            if defecto is _OBLIGATORIA:
                raise ErrorConfig(f"{ruta}.{clave}: falta (obligatoria)")
            salida[clave] = defecto
            continue
        valor = tabla[clave]
        if tipo is list:
            if not isinstance(valor, list) or not valor:
                raise ErrorConfig(f"{ruta}.{clave}: debe ser una lista no vacía de enteros")
            if not all(_tipo_ok(v, int) and v > 0 for v in valor):
                raise ErrorConfig(f"{ruta}.{clave}: todos los valores deben ser enteros > 0")
            if len(set(valor)) != len(valor):
                raise ErrorConfig(f"{ruta}.{clave}: valores repetidos")
            valor = tuple(valor)
        elif not _tipo_ok(valor, tipo):
            raise ErrorConfig(f"{ruta}.{clave}: se esperaba {tipo.__name__}, hay {type(valor).__name__}")
        elif tipo is float:
            valor = float(valor)
        salida[clave] = valor
    return salida


def _validar_url(url: str) -> None:
    partes = urlsplit(url)
    if partes.scheme not in ("http", "https") or not partes.hostname:
        raise ErrorConfig(f"servidor.url: {url!r} no es una URL http(s) válida")
    try:
        puerto = partes.port
    except ValueError as e:
        raise ErrorConfig(f"servidor.url: puerto inválido en {url!r}") from e
    # §3.2: en compute-0-0 el 8000 local lo ocupa un servicio ajeno (cedia-small).
    if puerto == 8000 and partes.hostname in ("127.0.0.1", "localhost", "::1"):
        raise ErrorConfig(
            "servidor.url: el puerto local 8000 lo ocupa cedia-small en compute-0-0; "
            "usa 8100 (vLLM) o 11434 (Ollama)"
        )


def desde_dict(datos: dict[str, Any]) -> Config:
    """Construye y valida la configuración a partir del TOML ya parseado."""
    secciones = ("experimento", "servidor", "carga", "telemetria", "salida")
    desconocidas = sorted(set(datos) - set(secciones))
    if desconocidas:
        raise ErrorConfig(f"secciones desconocidas: {', '.join(desconocidas)}")
    for s in ("experimento", "servidor", "carga"):
        if s not in datos:
            raise ErrorConfig(f"[{s}]: falta la sección (obligatoria)")

    exp = _leer_tabla(datos["experimento"], "experimento", {
        "nombre": (str, _OBLIGATORIA),
        "descripcion": (str, ""),
    })
    if not exp["nombre"].strip():
        raise ErrorConfig("experimento.nombre: no puede estar vacío")
    if any(c in exp["nombre"] for c in '/\\:*?"<>| '):
        raise ErrorConfig("experimento.nombre: se usa como nombre de carpeta; sin espacios ni /\\:*?\"<>|")

    srv = _leer_tabla(datos["servidor"], "servidor", {
        "backend": (str, _OBLIGATORIA),
        "url": (str, _OBLIGATORIA),
        "modelo": (str, _OBLIGATORIA),
        "gpu_esperada": (str, None),
    }, subtablas=("opciones_ollama",))
    if srv["backend"] not in BACKENDS:
        raise ErrorConfig(f"servidor.backend: {srv['backend']!r} no es uno de {BACKENDS}")
    _validar_url(srv["url"])

    opciones = None
    tabla_ollama = datos["servidor"].get("opciones_ollama")
    if srv["backend"] == "ollama":
        # §2: sin num_ctx Ollama trunca el prompt en silencio; sin keep_alive descarga el modelo.
        if tabla_ollama is None:
            raise ErrorConfig("[servidor.opciones_ollama]: obligatoria con backend = \"ollama\" (num_ctx, keep_alive)")
        o = _leer_tabla(tabla_ollama, "servidor.opciones_ollama", {
            "num_ctx": (int, _OBLIGATORIA),
            "keep_alive": (str, _OBLIGATORIA),
        })
        if o["num_ctx"] <= 0:
            raise ErrorConfig("servidor.opciones_ollama.num_ctx: debe ser > 0")
        opciones = OpcionesOllama(**o)
    elif tabla_ollama is not None:
        raise ErrorConfig("[servidor.opciones_ollama]: solo válida con backend = \"ollama\"")

    car = _leer_tabla(datos["carga"], "carga", {
        "entradas": (list, _OBLIGATORIA),
        "salidas": (list, _OBLIGATORIA),
        "concurrencias": (list, [1]),
        "repeticiones": (int, _OBLIGATORIA),
        "calentamiento": (int, 0),
        "modo": (str, "cerrado"),
        "tasa_rps": (float, None),
        "semilla": (int, _OBLIGATORIA),
        "temperatura": (float, 0.0),
    })
    car["concurrencias"] = tuple(car["concurrencias"])
    if car["repeticiones"] < 1:
        raise ErrorConfig("carga.repeticiones: debe ser >= 1")
    if car["calentamiento"] < 0:
        raise ErrorConfig("carga.calentamiento: debe ser >= 0")
    if car["temperatura"] < 0:
        raise ErrorConfig("carga.temperatura: debe ser >= 0")
    if car["modo"] not in MODOS_CARGA:
        raise ErrorConfig(f"carga.modo: {car['modo']!r} no es uno de {MODOS_CARGA}")
    if car["modo"] == "poisson":
        if car["tasa_rps"] is None or car["tasa_rps"] <= 0:
            raise ErrorConfig("carga.tasa_rps: obligatoria y > 0 con modo = \"poisson\"")
    elif car["tasa_rps"] is not None:
        raise ErrorConfig("carga.tasa_rps: solo válida con modo = \"poisson\"")

    tel = _leer_tabla(datos.get("telemetria", {}), "telemetria", {
        "activar": (bool, True),
        "intervalo_ms": (int, 500),
    })
    if tel["intervalo_ms"] < 100:
        raise ErrorConfig("telemetria.intervalo_ms: debe ser >= 100")

    sal = _leer_tabla(datos.get("salida", {}), "salida", {"dir": (str, "resultados")})
    if not sal["dir"].strip():
        raise ErrorConfig("salida.dir: no puede estar vacío")

    return Config(
        experimento=Experimento(**exp),
        servidor=Servidor(**srv, opciones_ollama=opciones),
        carga=Carga(**car),
        telemetria=Telemetria(**tel),
        salida=Salida(**sal),
    )


def cargar(ruta: Path) -> Config:
    """Lee y valida un TOML de experimento. Lanza ErrorConfig si no es válido."""
    try:
        with open(ruta, "rb") as f:
            datos = tomllib.load(f)
    except FileNotFoundError as e:
        raise ErrorConfig(f"{ruta}: no existe") from e
    except tomllib.TOMLDecodeError as e:
        raise ErrorConfig(f"{ruta}: TOML mal formado: {e}") from e
    return desde_dict(datos)
