"""Guardias + calibración + tanda secuencial + escritura de resultados (DESARROLLO.md §5, §6, §10).

Fase 1: backend Ollama, concurrencia 1. Orden fijo:

  G8 (directorio) → G6 (contexto) → G1 (salud) → G5 (modelo) → carga del modelo →
  G3 (Ollama en GPU) → G2 (memoria GPU) → G7 (calibración de la entrada) → tanda

La carga del modelo es una solicitud aparte ("Hola", 1 token) que se cronometra y se guarda en
`entorno.json`, igual que en `medir_ollama.py`; así `load_duration` no contamina la primera medida.

Cada tanda escribe en una carpeta nueva `resultados/<nombre>-<AAAAMMDD-HHMMSS>/` y nunca sobrescribe.
`datos.jsonl` se escribe solicitud a solicitud (si el job muere, lo medido queda); `datos.json`,
`resumen.json` y `entorno.json` se escriben al final, también si la tanda se interrumpe o falla
una guardia.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable

from medidor import __version__, conexion, guardias
from medidor.backends.base import Resultado, Solicitud, medir
from medidor.backends.ollama import TIMEOUT_CONEXION_S, BackendOllama
from medidor.config import Config
from medidor.esquema import ESQUEMA, escribir_json
from medidor.prompts import GeneradorPrompts, calibrar, dentro_de_tolerancia
from medidor.resumen import resumir

SALIDA_OK, SALIDA_GUARDIA, SALIDA_CONFIG, SALIDA_PARCIAL = 0, 1, 2, 3
MAX_ERRORES_SEGUIDOS = 3
DEPENDENCIAS = ("httpx", "matplotlib", "numpy")


def version_paquete(paquete: str) -> str | None:
    try:
        return metadata.version(paquete)
    except metadata.PackageNotFoundError:
        return None


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class Registro:
    """Mensajes con hora a la consola y, si hay carpeta de resultados, a `log.txt`."""

    def __init__(self, consola: Callable[[str], None] = print) -> None:
        self._consola = consola
        self._fichero = None

    def abrir(self, ruta: Path) -> None:
        self._fichero = open(ruta, "x", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        linea = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self._consola(linea)
        if self._fichero:
            self._fichero.write(linea + "\n")
            self._fichero.flush()

    def cerrar(self) -> None:
        if self._fichero:
            self._fichero.close()
            self._fichero = None


@dataclass
class Informe:
    codigo: int
    guardias: list[guardias.ResultadoGuardia] = field(default_factory=list)
    dir: Path | None = None
    mensaje: str = ""


def comprobar_fase(cfg: Config) -> str | None:
    """Lo que la fase 1 todavía no sabe hacer. None si la config se puede ejecutar."""
    if cfg.servidor.backend != "ollama":
        return f"backend {cfg.servidor.backend!r}: en la fase 1 solo está Ollama (vLLM llega en la fase 2)"
    if cfg.carga.concurrencias != (1,) or cfg.carga.modo != "cerrado":
        return "concurrencia > 1 o modo poisson: llegan en la fase 4 (ahora solo concurrencia = [1], modo cerrado)"
    return None


def _entrada_datos(i: int, cfg: Config, entrada: int, salida: int, rep: int, calentamiento: bool,
                   palabras: int, res: Resultado) -> dict:
    d = {
        "id": i,
        "punto": {"entrada": entrada, "salida": salida, "concurrencia": 1},
        "repeticion": rep,
        "calentamiento": calentamiento,
        "backend": cfg.servidor.backend,
        "modelo": cfg.servidor.modelo,
        "palabras_prompt": palabras,
        "tokens_entrada_pedidos": entrada,
        **res.a_dict(),
    }
    d["entrada_fuera_tolerancia"] = (res.tokens_entrada is not None
                                     and not dentro_de_tolerancia(res.tokens_entrada, entrada))
    return d


async def ejecutar(cfg: Config, ruta_config: Path, *, medir_tanda: bool,
                   base_salida: Path | None = None, consola: Callable[[str], None] = print) -> Informe:
    """`medir_tanda=False` es `medidor comprobar`: guardias y sondeos, sin escribir nada."""
    log = Registro(consola)
    inf = Informe(SALIDA_OK)

    problema = comprobar_fase(cfg)
    if problema:
        log(f"config no ejecutable todavía: {problema}")
        return Informe(SALIDA_CONFIG, mensaje=problema)

    srv, car, oll = cfg.servidor, cfg.carga, cfg.servidor.opciones_ollama
    assert oll is not None  # garantizado por config.py con backend = "ollama"
    base = (base_salida or Path.cwd()) / cfg.salida.dir

    def guardia(r: guardias.ResultadoGuardia) -> bool:
        inf.guardias.append(r)
        log(f"{r.id} {'OK   ' if r.ok else 'FALLO'} {r.mensaje}")
        return r.ok

    entorno: dict = {
        "medidor": __version__, "python": platform.python_version(), "python_ejecutable": sys.executable,
        "plataforma": platform.platform(),
        "dependencias": {p: version_paquete(p) for p in DEPENDENCIAS},
        **conexion.info_nodo(),
        "modo_cliente": "nodo" if shutil.which("nvidia-smi") else "tunel",
        "config": {"ruta": str(ruta_config), "experimento": cfg.experimento.nombre,
                   "descripcion": cfg.experimento.descripcion},
        "servidor": {"backend": srv.backend, "url": srv.url, "modelo": srv.modelo,
                     "num_ctx": oll.num_ctx, "keep_alive": oll.keep_alive, "log": oll.log},
        "tiempos_espera_s": {"solicitud_sin_datos": srv.timeout_s, "conexion": TIMEOUT_CONEXION_S,
                             "nvidia_smi": conexion.TIMEOUT_NVIDIA_SMI_S},
        "max_errores_seguidos": MAX_ERRORES_SEGUIDOS,
        "t_inicio_utc": _utc(), "loadavg_inicio": conexion.loadavg(),
        "gpu_inicio": conexion.instantanea_gpu(),
    }
    solicitudes: list[dict] = []
    estado = "sin empezar"

    if not guardia(guardias.g8_directorio(base)):
        return Informe(SALIDA_GUARDIA, inf.guardias, mensaje="G8")
    if not guardia(guardias.g6_contexto(oll.num_ctx, car.entradas, car.salidas)):
        return Informe(SALIDA_GUARDIA, inf.guardias, mensaje="G6")

    jsonl = None
    if medir_tanda:
        inf.dir = base / f"{cfg.experimento.nombre}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        inf.dir.mkdir(parents=True, exist_ok=False)      # nunca reutiliza una carpeta
        shutil.copyfile(ruta_config, inf.dir / "config.toml")
        log.abrir(inf.dir / "log.txt")
        jsonl = open(inf.dir / "datos.jsonl", "x", encoding="utf-8")
        log(f"resultados en {inf.dir}")

    try:
        async with BackendOllama(srv.url, srv.modelo, oll.num_ctx, oll.keep_alive, srv.timeout_s) as be:
            estado = "guardia fallida"
            if not guardia(await guardias.g1_salud(be)):
                return _fallo(inf, "G1")
            entorno["servidor"]["version"] = inf.guardias[-1].datos.get("version")
            if not guardia(await guardias.g5_modelo(be, srv.modelo)):
                return _fallo(inf, "G5")
            entorno["servidor"]["info_modelo"] = await be.info_modelo(srv.modelo)

            log(f"cargando el modelo (tiempo de espera sin datos: {srv.timeout_s:.0f} s)...")
            carga = await medir(be, Solicitud("Hola", 1, car.temperatura, car.semilla))
            entorno["carga_modelo"] = carga.a_dict()
            if carga.error:
                guardia(guardias.ResultadoGuardia("carga", False, f"la carga del modelo falló: {carga.error}"))
                return _fallo(inf, "carga")
            guardia(guardias.ResultadoGuardia(
                "carga", True, f"modelo cargado en {carga.e2e_s:.1f} s "
                f"(load_duration {carga.servidor.get('load_duration_s') or 0:.1f} s)"))

            if not guardia(await guardias.g3_ollama_gpu(be, srv.modelo, oll.log)):
                return _fallo(inf, "G3")
            if not guardia(guardias.g2_memoria_gpu()):
                return _fallo(inf, "G2")

            prompts = GeneradorPrompts(car.semilla)

            async def tokens_de(palabras: int) -> int:
                _, p = prompts.nuevo(palabras)
                r = await medir(be, Solicitud(p, 1, car.temperatura, car.semilla))
                if r.error or r.tokens_entrada is None:
                    raise RuntimeError(f"sondeo de {palabras} palabras falló: {r.error}")
                return r.tokens_entrada

            calibraciones = []
            try:
                for n in sorted(set(car.entradas)):
                    calibraciones.append(await calibrar(n, tokens_de))
            except RuntimeError as e:
                guardia(guardias.ResultadoGuardia("G7", False, str(e)))
                return _fallo(inf, "G7")
            if not guardia(guardias.g7_entrada(calibraciones)):
                return _fallo(inf, "G7")
            palabras_de = {c.objetivo: c.palabras for c in calibraciones}

            if not medir_tanda:
                estado = "comprobada"
                log("todas las guardias en verde (no se ha medido nada)")
                return inf

            # --- tanda -----------------------------------------------------------------------
            estado = "en curso"
            total = len(car.salidas) * len(car.entradas) * (car.calentamiento + car.repeticiones)
            seguidos = 0
            for salida in car.salidas:
                for entrada in car.entradas:
                    for k in range(car.calentamiento + car.repeticiones):
                        es_cal = k < car.calentamiento
                        rep = k - car.calentamiento if not es_cal else k
                        i, prompt = prompts.nuevo(palabras_de[entrada])
                        res = await medir(be, Solicitud(prompt, salida, car.temperatura, car.semilla))
                        d = _entrada_datos(i, cfg, entrada, salida, rep, es_cal, palabras_de[entrada], res)
                        solicitudes.append(d)
                        jsonl.write(json.dumps(d, ensure_ascii=False) + "\n")
                        jsonl.flush()
                        log(_linea_progreso(len(solicitudes), total, d))
                        seguidos = seguidos + 1 if res.error else 0
                        if seguidos >= MAX_ERRORES_SEGUIDOS:
                            estado = f"parada: {MAX_ERRORES_SEGUIDOS} errores seguidos"
                            log(f"PARADA: {MAX_ERRORES_SEGUIDOS} errores seguidos; último: {res.error}. "
                                "No se reintenta: revisa el log del servidor (medidor no mata nada).")
                            inf.codigo = SALIDA_PARCIAL
                            return inf
            con_error = sum(1 for s in solicitudes if s["error"] and not s["calentamiento"])
            estado = "completa" if not con_error else f"completa con {con_error} errores"
            if con_error:
                inf.codigo = SALIDA_PARCIAL
            log(f"tanda {estado}")
            return inf
    except KeyboardInterrupt:
        estado = "interrumpida (Ctrl+C)"
        raise
    except BaseException as e:
        estado = f"interrumpida: {type(e).__name__}: {e}"
        raise
    finally:
        if jsonl:
            jsonl.close()
        if inf.dir is not None:
            entorno.update({"estado": estado, "t_fin_utc": _utc(), "loadavg_fin": conexion.loadavg(),
                            "gpu_fin": conexion.instantanea_gpu(),
                            "guardias": [g.a_dict() for g in inf.guardias]})
            _escribir_resultados(inf.dir, cfg, entorno, solicitudes, estado)
            log(f"escritos entorno.json, datos.json y resumen.json en {inf.dir}")
        log.cerrar()


def _fallo(inf: Informe, id_guardia: str) -> Informe:
    inf.codigo = SALIDA_GUARDIA
    inf.mensaje = id_guardia
    return inf


def _linea_progreso(n: int, total: int, d: dict) -> str:
    p = d["punto"]
    marca = " (calentamiento)" if d["calentamiento"] else ""
    if d["error"]:
        return f"{n}/{total} in={p['entrada']} out={p['salida']}{marca} ERROR {d['error']}"
    srv = d["servidor"]
    return (f"{n}/{total} in={p['entrada']}→{d['tokens_entrada']} out={d['tokens_salida']}{marca} "
            f"ttft={d['ttft_s']:.3f} s tpot={1000 * (d['tpot_s'] or 0):.2f} ms "
            f"prefill={srv.get('prefill_tok_s') or 0:.0f} tok/s decode={srv.get('decode_tok_s') or 0:.1f} tok/s")


NOTAS_OLLAMA = [
    "Ollama no informa de los tokens por fragmento: itl_s son intervalos entre líneas NDJSON con texto.",
    "Ollama 0.24 no tiene ignore_eos: n_salida_corta cuenta las solicitudes que acabaron antes de lo pedido.",
    "servidor_prefill_tok_s = prompt_eval_count / prompt_eval_duration; servidor_decode_tok_s = eval_count / "
    "eval_duration (mismas definiciones que medir_ollama.py).",
    "Las solicitudes de calentamiento y las que tienen error no entran en los estadísticos.",
]


def cabecera(cfg: Config) -> dict:
    return {"experimento": cfg.experimento.nombre, "backend": cfg.servidor.backend, "modelo": cfg.servidor.modelo}


def resumen_desde_datos(datos: dict, estado: str | None = None) -> dict:
    return {
        "experimento": datos.get("experimento"), "backend": datos.get("backend"), "modelo": datos.get("modelo"),
        "estado": estado if estado is not None else datos.get("estado"),
        "generado_utc": _utc(),
        "notas": NOTAS_OLLAMA if datos.get("backend") == "ollama" else [],
        "puntos": resumir(datos["solicitudes"]),
    }


def _escribir_resultados(d: Path, cfg: Config, entorno: dict, solicitudes: list[dict], estado: str) -> None:
    escribir_json(d / "entorno.json", entorno)
    datos = {**cabecera(cfg), "estado": estado, "solicitudes": solicitudes}
    escribir_json(d / "datos.json", datos)
    escribir_json(d / "resumen.json", resumen_desde_datos(datos))
