import json

import pytest

from medidor.esquema import ESQUEMA, escribir_json


def test_escribe_esquema_primero(tmp_path):
    p = tmp_path / "datos.json"
    escribir_json(p, {"solicitudes": []})
    contenido = json.loads(p.read_text(encoding="utf-8"))
    assert list(contenido) == ["esquema", "solicitudes"]
    assert contenido["esquema"] == ESQUEMA


def test_no_sobrescribe(tmp_path):
    p = tmp_path / "datos.json"
    escribir_json(p, {"a": 1})
    with pytest.raises(FileExistsError):
        escribir_json(p, {"a": 2})
    assert json.loads(p.read_text(encoding="utf-8"))["a"] == 1


def test_rechaza_otro_esquema(tmp_path):
    with pytest.raises(ValueError):
        escribir_json(tmp_path / "x.json", {"esquema": "otro/v9"})
