import copy
from pathlib import Path

import pytest

from medidor import config
from medidor.config import ErrorConfig

RAIZ = Path(__file__).resolve().parent.parent

BASE = {
    "experimento": {"nombre": "prueba"},
    "servidor": {
        "backend": "ollama",
        "url": "http://127.0.0.1:11434",
        "modelo": "nemotron-3-nano:30b",
        "opciones_ollama": {"num_ctx": 8192, "keep_alive": "30m"},
    },
    "carga": {"entradas": [128, 512], "salidas": [128], "repeticiones": 3, "semilla": 1},
}


def con(**cambios):
    """Copia de BASE con cambios del tipo seccion__clave=valor (valor None borra la clave)."""
    d = copy.deepcopy(BASE)
    for ruta, valor in cambios.items():
        *padres, hoja = ruta.split("__")
        t = d
        for p in padres:
            t = t.setdefault(p, {})
        if valor is None:
            t.pop(hoja, None)
        else:
            t[hoja] = valor
    return d


@pytest.mark.parametrize("ruta", sorted((RAIZ / "configs").glob("*.toml")), ids=lambda p: p.name)
def test_configs_del_repo_son_validas(ruta):
    config.cargar(ruta)


def test_defectos():
    cfg = config.desde_dict(BASE)
    assert cfg.carga.concurrencias == (1,)
    assert cfg.carga.calentamiento == 0
    assert cfg.carga.modo == "cerrado"
    assert cfg.telemetria.activar is True and cfg.telemetria.intervalo_ms == 500
    assert cfg.salida.dir == "resultados"
    assert cfg.servidor.opciones_ollama == config.OpcionesOllama(num_ctx=8192, keep_alive="30m")


@pytest.mark.parametrize("cambios, mensaje", [
    ({"extra": {"a": 1}}, "secciones desconocidas: extra"),
    ({"carga__repeticone": 3}, "claves desconocidas: repeticone"),
    ({"servidor__opciones_ollama__numctx": 1}, "claves desconocidas: numctx"),
    ({"carga__repeticiones": True}, "se esperaba int, hay bool"),
    ({"carga__repeticiones": "10"}, "se esperaba int"),
    ({"carga__repeticiones": 0}, "repeticiones"),
    ({"carga__semilla": None}, "carga.semilla: falta"),
    ({"carga__entradas": []}, "lista no vacía"),
    ({"carga__entradas": [128, -1]}, "enteros > 0"),
    ({"carga__entradas": [128, 128]}, "repetidos"),
    ({"servidor__backend": "tgi"}, "servidor.backend"),
    ({"servidor__url": "127.0.0.1:11434"}, "URL http(s)"),
    ({"servidor__url": "http://localhost:8000"}, "cedia-small"),
    ({"servidor__opciones_ollama": None}, "obligatoria con backend"),
    ({"servidor__opciones_ollama__keep_alive": None}, "keep_alive: falta"),
    ({"servidor__backend": "vllm"}, "solo válida con backend"),
    ({"carga__modo": "poisson"}, "tasa_rps: obligatoria"),
    ({"carga__tasa_rps": 2.0}, "solo válida con modo"),
    ({"carga__modo": "abierto"}, "carga.modo"),
    ({"experimento__nombre": "con espacio"}, "nombre de carpeta"),
    ({"telemetria": {"intervalo_ms": 10}}, "intervalo_ms"),
])
def test_errores(cambios, mensaje):
    with pytest.raises(ErrorConfig) as e:
        config.desde_dict(con(**cambios))
    assert mensaje in str(e.value)


def test_poisson_valido():
    cfg = config.desde_dict(con(carga__modo="poisson", carga__tasa_rps=2))
    assert cfg.carga.tasa_rps == 2.0 and isinstance(cfg.carga.tasa_rps, float)


def test_vllm_sin_opciones_ollama():
    d = con(servidor__backend="vllm", servidor__url="http://127.0.0.1:8100")
    del d["servidor"]["opciones_ollama"]
    assert config.desde_dict(d).servidor.opciones_ollama is None


def test_toml_mal_formado(tmp_path):
    p = tmp_path / "x.toml"
    p.write_text("[carga\n", encoding="utf-8")
    with pytest.raises(ErrorConfig, match="TOML mal formado"):
        config.cargar(p)


def test_fichero_inexistente(tmp_path):
    with pytest.raises(ErrorConfig, match="no existe"):
        config.cargar(tmp_path / "no.toml")
