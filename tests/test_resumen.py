import pytest

from medidor.resumen import estadisticos, resumir


def test_estadisticos():
    e = estadisticos([1.0, 2.0, 3.0, 4.0, None])
    assert e["n"] == 4 and e["media"] == 2.5 and e["min"] == 1.0 and e["max"] == 4.0
    assert e["p50"] == 2.5 and e["desv"] == pytest.approx(1.2909944)
    assert estadisticos([None]) is None
    assert estadisticos([5.0])["desv"] == 0.0


def _sol(entrada, ttft, cal=False, error=None, tokens_salida=128):
    return {"punto": {"entrada": entrada, "salida": 128, "concurrencia": 1}, "calentamiento": cal,
            "error": error, "ttft_s": ttft, "itl_s": [0.01, 0.02], "tokens_salida": tokens_salida,
            "finish_reason": "length", "servidor": {"prefill_tok_s": 1000.0}}


def test_resumir_excluye_calentamiento_y_errores_pero_los_cuenta():
    sols = [_sol(170, 9.0, cal=True), _sol(170, 1.0), _sol(170, 3.0), _sol(170, None, error="x"),
            _sol(615, 2.0, tokens_salida=100)]
    a, b = resumir(sols)
    assert a["punto"] == {"entrada": 170, "salida": 128, "concurrencia": 1}
    assert (a["n_solicitudes"], a["n_ok"], a["n_errores"], a["n_calentamiento"]) == (3, 2, 1, 1)
    assert a["errores"] == ["x"]
    assert a["ttft_s"]["media"] == 2.0 and a["ttft_s"]["n"] == 2     # el 9.0 de calentamiento no entra
    assert a["itl_s"]["n"] == 4
    assert a["servidor_prefill_tok_s"]["media"] == 1000.0
    assert a["servidor_decode_tok_s"] is None                         # no hay datos: no se inventa
    assert b["n_salida_corta"] == 1 and a["n_salida_corta"] == 0
