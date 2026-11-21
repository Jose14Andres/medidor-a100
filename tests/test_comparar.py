import json

import pytest

from medidor import comparar
from medidor.cli import SALIDA_CONFIG, SALIDA_NO_PASA, SALIDA_OK, main
from medidor.esquema import ESQUEMA

# Medianas del job 36268 (FASE-1.md, tabla de la referencia).
REF_36268 = [(128, 170, 1513.6, 152.0, 0.983), (512, 614.8, 2204.8, 153.4, 1.599),
             (2048, 2389.8, 3007.2, 152.1, 2.554)]


def _est(valor, clave):
    return None if valor is None else {"n": 5, clave: valor}


def resumen_referencia(puntos=REF_36268) -> dict:
    """Formato de `medir_ollama.py`: sin `esquema`, estadístico `mediana`."""
    return {
        "meta": {"modelo": "nemotron-3-nano:30b", "slurm_job": "36268"},
        "puntos": [{
            "entrada_palabras": pal, "fallos": 0,
            "prompt_tokens": {"n": 5, "media": tok, "mediana": tok},
            "servidor_prefill_tok_s": _est(pre, "mediana"),
            "servidor_decode_tok_s": _est(dec, "mediana"),
            "ttft_s": _est(ttft, "mediana"),
            "tpot_cliente_s": _est(0.007, "mediana"),
        } for pal, tok, pre, dec, ttft in puntos],
    }


def punto_medidor(entrada, tokens, prefill, decode, ttft=0.2, tpot=0.0066) -> dict:
    """Formato del medidor (`resumen.puntos[]`): estadístico `p50`."""
    return {
        "punto": {"entrada": entrada, "salida": 128, "concurrencia": 1},
        "n_ok": 5, "n_errores": 0, "n_salida_corta": 0, "n_entrada_fuera_tolerancia": 0,
        "tokens_entrada": _est(tokens, "media"),
        "servidor_prefill_tok_s": _est(prefill, "p50"),
        "servidor_decode_tok_s": _est(decode, "p50"),
        "ttft_s": _est(ttft, "p50"),
        "tpot_s": _est(tpot, "p50"),
    }


def igual_que_referencia() -> list[dict]:
    return [punto_medidor(e, tok, pre, dec) for e, tok, pre, dec in
            [(170, 170, 1513.6, 152.0), (615, 615, 2204.8, 153.4), (2390, 2390, 3007.2, 152.1)]]


@pytest.fixture
def entradas(tmp_path):
    """Escribe una tanda y una referencia; devuelve (dir_tanda, ruta_referencia, escribir)."""
    tanda = tmp_path / "v1-validacion-36268-20261121-100000"
    tanda.mkdir()
    ref = tmp_path / "resumen_36268.json"

    def escribir(puntos, referencia=None, esquema=ESQUEMA):
        (tanda / "resumen.json").write_text(json.dumps({"esquema": esquema, "puntos": puntos}), encoding="utf-8")
        ref.write_text(json.dumps(referencia or resumen_referencia()), encoding="utf-8")
        return tanda, ref

    return escribir


def _correr(tanda, ref) -> int:
    return main(["comparar", str(tanda), "--referencia", str(ref)])


def test_todo_pasa(entradas, capsys):
    tanda, ref = entradas(igual_que_referencia())
    assert _correr(tanda, ref) == SALIDA_OK
    salida = capsys.readouterr().out
    assert "resultado: PASA" in salida and "NO PASA" not in salida
    res = json.loads((tanda / "comparacion.json").read_text(encoding="utf-8"))
    assert res["esquema"] == ESQUEMA and res["pasa"] is True
    assert len(res["parejas"]) == 3 and res["referencia_sin_pareja"] == [] and res["avisos"] == []
    assert res["referencia_meta"]["slurm_job"] == "36268"


@pytest.mark.parametrize("factor", [1.05, 0.95])
def test_borde_del_5_por_ciento_pasa(entradas, factor):
    ref = resumen_referencia([(128, 170, 1000.0, 100.0, 1.0)])
    tanda, r = entradas([punto_medidor(170, 170, 1000.0 * factor, 100.0 * factor)], ref)
    assert _correr(tanda, r) == SALIDA_OK


@pytest.mark.parametrize("factor", [1.0501, 0.9499])
def test_justo_fuera_del_borde_no_pasa(entradas, factor, capsys):
    ref = resumen_referencia([(128, 170, 1000.0, 100.0, 1.0)])
    tanda, r = entradas([punto_medidor(170, 170, 1000.0 * factor, 100.0)], ref)
    assert _correr(tanda, r) == SALIDA_NO_PASA
    assert "resultado: NO PASA" in capsys.readouterr().out
    res = json.loads((tanda / "comparacion.json").read_text(encoding="utf-8"))
    c = res["parejas"][0]["criterio"]
    assert c["servidor_prefill_tok_s"]["pasa"] is False and c["servidor_decode_tok_s"]["pasa"] is True
    assert res["parejas"][0]["pasa"] is False


def test_informativo_no_entra_en_el_criterio(entradas):
    puntos = igual_que_referencia()
    for p in puntos:
        p["ttft_s"]["p50"] = 10.0        # TTFT del cliente muy distinto: solo se informa
    tanda, ref = entradas(puntos)
    assert _correr(tanda, ref) == SALIDA_OK
    inf = json.loads((tanda / "comparacion.json").read_text(encoding="utf-8"))["parejas"][0]["informativo"]
    assert inf["ttft_cliente_s"]["referencia"] == 0.983 and inf["ttft_cliente_s"]["nuevo"] == 10.0
    assert "tpot_cliente_s" in inf


def test_tokens_fuera_del_2_por_ciento_solo_avisa(entradas, capsys):
    puntos = igual_que_referencia()
    puntos[0]["tokens_entrada"]["media"] = 175          # +2,9 % frente a 170
    tanda, ref = entradas(puntos)
    assert _correr(tanda, ref) == SALIDA_OK
    assert "aviso:" in capsys.readouterr().out
    res = json.loads((tanda / "comparacion.json").read_text(encoding="utf-8"))
    assert res["parejas"][0]["aviso_tokens"] is True and res["parejas"][1]["aviso_tokens"] is False


def test_referencia_sin_pareja_no_pasa_y_empareja_uno_a_uno(entradas):
    # Sin el punto de 615: el 614,8 de la referencia no puede reutilizar el 170 ya emparejado.
    puntos = [p for p in igual_que_referencia() if p["punto"]["entrada"] != 615]
    tanda, ref = entradas(puntos)
    assert _correr(tanda, ref) == SALIDA_NO_PASA
    res = json.loads((tanda / "comparacion.json").read_text(encoding="utf-8"))
    assert [p["referencia"]["entrada_palabras"] for p in res["parejas"]] == [128, 2048]
    assert [r["entrada_palabras"] for r in res["referencia_sin_pareja"]] == [512]
    assert all(p["pasa"] for p in res["parejas"])


def test_punto_nuevo_sin_pareja_solo_informa(entradas, capsys):
    puntos = igual_que_referencia() + [punto_medidor(4096, 4096, 3100.0, 150.0)]
    tanda, ref = entradas(puntos)
    assert _correr(tanda, ref) == SALIDA_OK
    assert "info: el punto 4096" in capsys.readouterr().out
    res = json.loads((tanda / "comparacion.json").read_text(encoding="utf-8"))
    assert res["nuevo_sin_pareja"] == [{"punto": {"entrada": 4096, "salida": 128, "concurrencia": 1},
                                        "tokens": 4096}]


def test_punto_sin_valores_no_pasa(entradas):
    puntos = igual_que_referencia()
    puntos[2] = punto_medidor(2390, None, None, None, ttft=None, tpot=None)   # todas con error
    puntos[2].update(n_ok=0, n_errores=5)
    tanda, ref = entradas(puntos)
    assert _correr(tanda, ref) == SALIDA_NO_PASA
    p = json.loads((tanda / "comparacion.json").read_text(encoding="utf-8"))["parejas"][2]
    assert p["nuevo"]["tokens_medidos"] is False and p["nuevo"]["tokens"] == 2390   # por tokens pedidos
    assert p["criterio"]["servidor_prefill_tok_s"]["motivo"] == "sin valor en la tanda"
    assert p["nuevo"]["diagnostico"]["n_errores"] == 5


def test_esquema_incorrecto(entradas, capsys):
    tanda, ref = entradas(igual_que_referencia(), esquema="medidor-a100/v0")
    assert _correr(tanda, ref) == SALIDA_CONFIG
    assert "esquema" in capsys.readouterr().err
    assert not (tanda / "comparacion.json").exists()


def test_referencia_con_otro_formato(entradas, tmp_path):
    tanda, ref = entradas(igual_que_referencia())
    ref.write_text(json.dumps({"esquema": ESQUEMA, "puntos": igual_que_referencia()}), encoding="utf-8")
    assert _correr(tanda, ref) == SALIDA_CONFIG
    assert _correr(tanda, tmp_path / "no-existe.json") == SALIDA_CONFIG


def test_no_sobrescribe(entradas):
    tanda, ref = entradas(igual_que_referencia())
    previo = tanda / "comparacion.json"
    previo.write_text("contenido previo\n", encoding="utf-8")
    assert _correr(tanda, ref) == SALIDA_OK
    assert _correr(tanda, ref) == SALIDA_OK          # mismo segundo: sufijo -2
    assert previo.read_text(encoding="utf-8") == "contenido previo\n"
    nuevos = sorted(p for p in tanda.glob("comparacion-*.json"))
    assert len(nuevos) == 2
    assert all(json.loads(p.read_text(encoding="utf-8"))["esquema"] == ESQUEMA for p in nuevos)


def test_emparejar_por_cercania():
    ref = [{"tokens": 170}, {"tokens": 614.8}, {"tokens": 2389.8}]
    nuevos = [{"tokens": 2390}, {"tokens": 171}, {"tokens": 600}]
    parejas, sin_r, sin_n = comparar.emparejar(ref, nuevos)
    assert parejas == [(0, 1), (1, 2), (2, 0)] and sin_r == [] and sin_n == []
