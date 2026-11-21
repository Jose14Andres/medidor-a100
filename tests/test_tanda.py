"""De punta a punta: `medidor comprobar | medir | resumir` contra el servidor falso."""

import json
from pathlib import Path

import pytest

from conftest import DATOS
from medidor import guardias
from medidor.cli import SALIDA_CONFIG, SALIDA_GUARDIA, SALIDA_OK, SALIDA_PARCIAL, main
from medidor.esquema import ESQUEMA
from servidor_falso import Ajustes, arrancar


def _config(tmp_path: Path, url: str, **extra) -> Path:
    log = tmp_path / "ollama_serve.log"
    if not log.exists():
        log.write_text((DATOS / "ollama_serve_gpu.log").read_text())
    car = {"entradas": "[20, 60]", "salidas": "[8]", "repeticiones": "3", "calentamiento": "1",
           "concurrencias": "[1]", "modo": '"cerrado"'}
    car.update(extra.pop("carga", {}))
    backend = extra.pop("backend", "ollama")
    texto = f'''
[experimento]
nombre = "prueba"
[servidor]
backend = "{backend}"
url = "{url}"
modelo = "m"
timeout_s = 10
[servidor.opciones_ollama]
num_ctx = 4096
keep_alive = "30m"
log = "{log}"
[carga]
{chr(10).join(f"{k} = {v}" for k, v in car.items())}
semilla = 1
[salida]
dir = "{tmp_path / 'resultados'}"
'''
    if backend != "ollama":
        texto = texto.replace("[servidor.opciones_ollama]\nnum_ctx = 4096\nkeep_alive = \"30m\"\n"
                              f"log = \"{log}\"\n", "")
    ruta = tmp_path / "x.toml"
    ruta.write_text(texto, encoding="utf-8")
    return ruta


@pytest.fixture
def entorno_pruebas(monkeypatch, tmp_path, gpu_falsa):
    monkeypatch.setattr(guardias, "RAICES_PERMITIDAS", (tmp_path,))
    return gpu_falsa


def _tandas(tmp_path):
    return sorted((tmp_path / "resultados").glob("prueba-*"))


def _leer(d, nombre):
    return json.loads((d / nombre).read_text(encoding="utf-8"))


def test_medir_completa(tmp_path, entorno_pruebas, capsys):
    with arrancar(Ajustes(modelo="m", ttft_s=0.02, itl_s=0.005)) as url:
        assert main(["medir", "-c", str(_config(tmp_path, url))]) == SALIDA_OK
    [d] = _tandas(tmp_path)
    assert {p.name for p in d.iterdir()} == {"config.toml", "log.txt", "datos.jsonl", "datos.json",
                                             "entorno.json", "resumen.json"}
    assert (d / "config.toml").read_text() == (tmp_path / "x.toml").read_text()

    ent, datos, res = _leer(d, "entorno.json"), _leer(d, "datos.json"), _leer(d, "resumen.json")
    assert ent["esquema"] == datos["esquema"] == res["esquema"] == ESQUEMA
    assert ent["estado"] == datos["estado"] == "completa"
    assert [g["id"] for g in ent["guardias"]] == ["G8", "G6", "G1", "G5", "carga", "G3", "G2", "G7"]
    assert all(g["ok"] for g in ent["guardias"])
    assert ent["tiempos_espera_s"]["solicitud_sin_datos"] == 10.0
    assert ent["carga_modelo"]["tokens_salida"] == 1

    sols = datos["solicitudes"]
    assert len(sols) == 2 * (1 + 3)
    assert len((d / "datos.jsonl").read_text().splitlines()) == len(sols)
    assert sum(s["calentamiento"] for s in sols) == 2
    assert len({s["id"] for s in sols}) == len(sols)
    for s in sols:
        assert s["error"] is None and not s["entrada_fuera_tolerancia"]
        assert abs(s["tokens_entrada"] - s["tokens_entrada_pedidos"]) <= 1

    a, b = res["puntos"]
    assert a["punto"] == {"entrada": 20, "salida": 8, "concurrencia": 1}
    assert (a["n_solicitudes"], a["n_ok"], a["n_errores"]) == (3, 3, 0)
    assert a["servidor_decode_tok_s"]["media"] == pytest.approx(8 / (0.005 * 7))
    assert "resultados:" in capsys.readouterr().out


def test_no_sobrescribe_dos_tandas(tmp_path, entorno_pruebas, monkeypatch):
    import medidor.tanda as tanda

    marcas = iter(["20261120-100000", "20261120-100001"])

    class Reloj:
        @staticmethod
        def now(tz=None):
            class M:
                def strftime(self, _):
                    return next(marcas)
                def isoformat(self, **k):
                    return "2026-11-20T10:00:00+00:00"
            return M()

    monkeypatch.setattr(tanda, "datetime", Reloj)
    with arrancar(Ajustes(modelo="m", ttft_s=0.0, itl_s=0.0)) as url:
        cfg = _config(tmp_path, url, carga={"entradas": "[20]", "repeticiones": "1", "calentamiento": "0"})
        assert main(["medir", "-c", str(cfg)]) == SALIDA_OK
        assert main(["medir", "-c", str(cfg)]) == SALIDA_OK
    assert [d.name for d in _tandas(tmp_path)] == ["prueba-20261120-100000", "prueba-20261120-100001"]


def test_comprobar_no_escribe_nada(tmp_path, entorno_pruebas):
    with arrancar(Ajustes(modelo="m", ttft_s=0.0, itl_s=0.0)) as url:
        assert main(["comprobar", "-c", str(_config(tmp_path, url))]) == SALIDA_OK
    assert not (tmp_path / "resultados").exists()


def test_guardia_g3_para_y_deja_constancia(tmp_path, entorno_pruebas):
    cfg = _config(tmp_path, "http://x")
    (tmp_path / "ollama_serve.log").write_text((DATOS / "ollama_serve_cpu.log").read_text())
    with arrancar(Ajustes(modelo="m", size_vram=0)) as url:
        cfg.write_text(cfg.read_text().replace("http://x", url))
        assert main(["medir", "-c", str(cfg)]) == SALIDA_GUARDIA
    [d] = _tandas(tmp_path)
    ent = _leer(d, "entorno.json")
    assert ent["estado"] == "guardia fallida"
    assert ent["guardias"][-1]["id"] == "G3" and not ent["guardias"][-1]["ok"]
    assert _leer(d, "datos.json")["solicitudes"] == []


def test_guardia_g2_gpu_vacia(tmp_path, entorno_pruebas):
    entorno_pruebas["usado_mib"] = 0
    with arrancar(Ajustes(modelo="m", ttft_s=0.0)) as url:
        assert main(["comprobar", "-c", str(_config(tmp_path, url))]) == SALIDA_GUARDIA


def test_guardia_g8_no_crea_carpeta(tmp_path, gpu_falsa):
    # Sin parchear RAICES_PERMITIDAS: tmp_path no está en /home.
    if any(r in tmp_path.resolve().parents for r in guardias.RAICES_PERMITIDAS):
        pytest.skip("tmp_path está bajo /home")
    with arrancar(Ajustes(modelo="m")) as url:
        assert main(["medir", "-c", str(_config(tmp_path, url))]) == SALIDA_GUARDIA
    assert not (tmp_path / "resultados").exists()


def test_para_tras_errores_seguidos(tmp_path, entorno_pruebas):
    with arrancar(Ajustes(modelo="m", ttft_s=0.0, itl_s=0.0, fallar_con_num_predict=8)) as url:
        assert main(["medir", "-c", str(_config(tmp_path, url))]) == SALIDA_PARCIAL
    [d] = _tandas(tmp_path)
    datos = _leer(d, "datos.json")
    assert datos["estado"] == "parada: 3 errores seguidos"
    assert len(datos["solicitudes"]) == 3
    assert all("HTTP 500" in s["error"] for s in datos["solicitudes"])
    assert _leer(d, "resumen.json")["puntos"][0]["n_errores"] == 2   # 1 era de calentamiento


def test_fases_futuras_son_error_de_config(tmp_path, entorno_pruebas):
    assert main(["medir", "-c", str(_config(tmp_path, "http://127.0.0.1:8100", backend="vllm"))]) == SALIDA_CONFIG
    cfg = _config(tmp_path, "http://127.0.0.1:11434", carga={"concurrencias": "[1, 4]"})
    assert main(["comprobar", "-c", str(cfg)]) == SALIDA_CONFIG


def test_resumir_no_sobrescribe(tmp_path, entorno_pruebas):
    with arrancar(Ajustes(modelo="m", ttft_s=0.0, itl_s=0.0)) as url:
        cfg = _config(tmp_path, url, carga={"entradas": "[20]", "repeticiones": "2", "calentamiento": "0"})
        assert main(["medir", "-c", str(cfg)]) == SALIDA_OK
    [d] = _tandas(tmp_path)
    original = (d / "resumen.json").read_text()
    assert main(["resumir", str(d)]) == SALIDA_OK
    assert (d / "resumen.json").read_text() == original
    [nuevo] = d.glob("resumen-*.json")
    assert _leer(d, nuevo.name)["puntos"] == _leer(d, "resumen.json")["puntos"]
