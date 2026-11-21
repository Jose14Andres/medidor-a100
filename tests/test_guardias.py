import asyncio
from pathlib import Path

import pytest

from conftest import DATOS
from medidor import conexion, guardias
from medidor.backends.ollama import BackendOllama
from medidor.prompts import Calibracion, calibrar
from servidor_falso import Ajustes, arrancar


def _con_backend(aj, fn, num_ctx=4096):
    with arrancar(aj) as url:
        async def corrida():
            async with BackendOllama(url, aj.modelo, num_ctx, "30m", 10) as be:
                return await fn(be)
        return asyncio.run(corrida())


# --- análisis del log real ---------------------------------------------------------------------

def test_log_real_gpu_36268():
    a = guardias.analizar_log_ollama((DATOS / "ollama_serve_gpu.log").read_text())
    assert a == {"librerias": ["CUDA"], "fallos_descubrimiento": 0, "offload": [53, 53]}


def test_log_real_cpu_36509():
    a = guardias.analizar_log_ollama((DATOS / "ollama_serve_cpu.log").read_text())
    assert a == {"librerias": ["cpu"], "fallos_descubrimiento": 1, "offload": [0, 49]}


def test_log_solo_cuenta_el_ultimo_arranque():
    texto = (DATOS / "ollama_serve_gpu.log").read_text() + (DATOS / "ollama_serve_cpu.log").read_text()
    assert guardias.analizar_log_ollama(texto)["librerias"] == ["cpu"]


# --- G3 ----------------------------------------------------------------------------------------

def _g3(aj, log):
    return _con_backend(aj, lambda be: guardias.g3_ollama_gpu(be, aj.modelo, str(log)))


def test_g3_ok_con_log_gpu():
    r = _g3(Ajustes(modelo="m"), DATOS / "ollama_serve_gpu.log")
    assert r.ok, r.mensaje
    assert "53/53" in r.mensaje


def test_g3_detecta_caida_a_cpu_del_17_nov():
    r = _g3(Ajustes(modelo="m", size_vram=0), DATOS / "ollama_serve_cpu.log")
    assert not r.ok
    assert "CUDA" in r.mensaje and "0/49" in r.mensaje and "size_vram = 0" in r.mensaje


def test_g3_detecta_carga_parcial_en_vram():
    r = _g3(Ajustes(modelo="m", size_vram=10, size=20), DATOS / "ollama_serve_gpu.log")
    assert not r.ok and "50%" in r.mensaje


def test_g3_sin_log(tmp_path):
    r = _g3(Ajustes(modelo="m"), tmp_path / "no-existe.log")
    assert not r.ok and "no se puede leer" in r.mensaje


def test_g3_expande_job(tmp_path, monkeypatch):
    (tmp_path / "ollama_serve_123.log").write_text((DATOS / "ollama_serve_gpu.log").read_text())
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert _g3(Ajustes(modelo="m"), tmp_path / "ollama_serve_{job}.log").ok
    monkeypatch.delenv("SLURM_JOB_ID")
    r = _g3(Ajustes(modelo="m"), tmp_path / "ollama_serve_{job}.log")
    assert not r.ok and "SLURM_JOB_ID" in r.mensaje


# --- G1, G2, G5, G6, G7, G8 --------------------------------------------------------------------

def test_g1_servidor_no_listo():
    be = BackendOllama("http://127.0.0.1:1", "m", 4096, "30m", 5)

    async def corrida():
        async with be:
            return await guardias.g1_salud(be)
    r = asyncio.run(corrida())
    assert not r.ok and "servidor no listo" in r.mensaje


def test_g1_y_g5():
    aj = Ajustes(modelo="m")
    assert _con_backend(aj, guardias.g1_salud).datos["version"] == "0.24.0"
    assert _con_backend(aj, lambda be: guardias.g5_modelo(be, "m")).ok
    r = _con_backend(aj, lambda be: guardias.g5_modelo(be, "otro"))
    assert not r.ok and "hay: m" in r.mensaje


def test_g2():
    gpu = lambda mib: (lambda: [{"usado_mib": mib}])
    assert guardias.g2_memoria_gpu(gpu(24819)).ok
    assert "no está en GPU" in guardias.g2_memoria_gpu(gpu(0)).mensaje

    def sin_nvidia_smi():
        raise conexion.ErrorConexion("nvidia-smi no está en el PATH")
    r = guardias.g2_memoria_gpu(sin_nvidia_smi)
    assert not r.ok and "nvidia-smi" in r.mensaje


def test_g6():
    assert guardias.g6_contexto(8192, [170, 615, 2390], [128]).ok
    r = guardias.g6_contexto(2048, [2048], [128])
    assert not r.ok and "truncaría" in r.mensaje


def test_g7_detecta_truncado_real_del_servidor():
    # num_ctx = 100: el servidor falso, como Ollama 0.24, trunca el prompt en silencio.
    from medidor.backends.base import Solicitud, medir
    from medidor.prompts import GeneradorPrompts

    aj = Ajustes(modelo="m")
    g = GeneradorPrompts(1)

    async def fn(be):
        async def tokens(p):
            return (await medir(be, Solicitud(g.nuevo(p)[1], 1))).tokens_entrada
        return [await calibrar(50, tokens), await calibrar(170, tokens)]

    cal = _con_backend(aj, fn, num_ctx=100)
    r = guardias.g7_entrada(cal)
    assert not r.ok and "pedidos 170 → reales 100" in r.mensaje
    assert [p["ok"] for p in r.datos["puntos"]] == [True, False]
    assert guardias.g7_entrada([Calibracion(170, 164, 170)]).ok


def test_g8(monkeypatch, tmp_path):
    assert guardias.g8_directorio(Path("/home/usuario/resultados")).ok
    r = guardias.g8_directorio(Path("/scratch/usuario/37560/resultados"))
    assert not r.ok and "/scratch" in r.mensaje
    assert not guardias.g8_directorio(Path("/homero/x")).ok    # prefijo de texto, no de ruta
    monkeypatch.setattr(guardias, "RAICES_PERMITIDAS", (tmp_path,))
    assert guardias.g8_directorio(tmp_path / "resultados").ok
