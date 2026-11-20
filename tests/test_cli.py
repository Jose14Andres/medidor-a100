import json
from pathlib import Path

from medidor import __version__
from medidor.cli import SALIDA_CONFIG, SALIDA_OK, main
from medidor.esquema import ESQUEMA

RAIZ = Path(__file__).resolve().parent.parent
CONFIG_OLLAMA = RAIZ / "configs" / "e1-prefill-ollama-nemotron.toml"


def test_entorno_sin_config(capsys):
    assert main(["entorno"]) == SALIDA_OK
    salida = capsys.readouterr().out
    assert __version__ in salida and "python" in salida


def test_entorno_json(capsys):
    assert main(["entorno", "--json", "-c", str(CONFIG_OLLAMA)]) == SALIDA_OK
    info = json.loads(capsys.readouterr().out)
    assert info["esquema"] == ESQUEMA
    assert info["config"]["valida"] is True
    assert info["config"]["backend"] == "ollama"
    assert set(info["dependencias"]) == {"httpx", "matplotlib", "numpy"}


def test_entorno_config_invalida(tmp_path, capsys):
    p = tmp_path / "mala.toml"
    p.write_text('[experimento]\nnombre = "x"\nerrata = 1\n', encoding="utf-8")
    assert main(["entorno", "-c", str(p)]) == SALIDA_CONFIG
    assert "NO VÁLIDA" in capsys.readouterr().out
