"""El servidor falso es la base de los tests de las fases siguientes: comprobar que imita bien."""

import json
import time

from servidor_falso import Ajustes, arrancar

PROMPT_8 = "a b c d e f g h"


def test_vllm_rutas_basicas(cliente):
    with arrancar(Ajustes(modelo="m")) as url:
        assert cliente.get(f"{url}/health").status_code == 200
        modelos = cliente.get(f"{url}/v1/models").json()["data"]
        assert [m["id"] for m in modelos] == ["m"]


def test_vllm_stream_sse(cliente):
    aj = Ajustes(modelo="m", ttft_s=0.1, itl_s=0.01)
    cuerpo = {"model": "m", "prompt": PROMPT_8, "max_tokens": 5, "stream": True,
              "stream_options": {"include_usage": True}}
    with arrancar(aj) as url:
        t0 = time.perf_counter()
        eventos, t_primero = [], None
        with cliente.stream("POST", f"{url}/v1/completions", json=cuerpo) as r:
            assert r.status_code == 200
            for linea in r.iter_lines():
                if not linea.startswith("data: "):
                    continue
                if t_primero is None:
                    t_primero = time.perf_counter() - t0
                eventos.append(linea[len("data: "):])
    assert eventos[-1] == "[DONE]"
    datos = [json.loads(e) for e in eventos[:-1]]
    textos = [d["choices"][0] for d in datos if d["choices"]]
    assert len(textos) == 5
    assert textos[-1]["finish_reason"] == "length"
    assert datos[-1]["usage"] == {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13}
    assert t_primero >= 0.1


def test_vllm_supera_max_model_len(cliente):
    with arrancar(Ajustes(modelo="m", max_model_len=10)) as url:
        r = cliente.post(f"{url}/v1/completions",
                         json={"model": "m", "prompt": PROMPT_8, "max_tokens": 5, "stream": True})
    assert r.status_code == 400


def test_ollama_rutas_basicas(cliente):
    with arrancar(Ajustes(modelo="m", size_vram=0)) as url:
        assert cliente.get(f"{url}/api/version").json()["version"] == "0.24.0"
        assert cliente.get(f"{url}/api/tags").json()["models"][0]["name"] == "m"
        assert cliente.get(f"{url}/api/ps").json()["models"][0]["size_vram"] == 0


def _generate(cliente, url, **extra):
    cuerpo = {"model": "m", "prompt": PROMPT_8, "raw": True,
              "options": {"num_predict": 4, "num_ctx": 4096}, **extra}
    with cliente.stream("POST", f"{url}/api/generate", json=cuerpo) as r:
        assert r.status_code == 200
        return [json.loads(l) for l in r.iter_lines() if l]


def test_ollama_stream_ndjson(cliente):
    with arrancar(Ajustes(modelo="m", ttft_s=0.02, itl_s=0.01, tokens_por_fragmento=3)) as url:
        lineas = _generate(cliente, url)
    trozos, final = lineas[:-1], lineas[-1]
    assert [t["response"] for t in trozos] == ["tok tok tok ", "tok "]
    assert final["done"] is True and final["done_reason"] == "length"
    assert final["prompt_eval_count"] == 8 and final["eval_count"] == 4
    assert final["prompt_eval_duration"] == 20_000_000


def test_ollama_trunca_en_silencio_sin_num_ctx(cliente):
    # El comportamiento que la guardia G7 tendrá que detectar.
    with arrancar(Ajustes(modelo="m", num_ctx_defecto=5)) as url:
        final = _generate(cliente, url, options={"num_predict": 2})[-1]
    assert final["prompt_eval_count"] == 5


def test_modelo_desconocido(cliente):
    with arrancar(Ajustes(modelo="m")) as url:
        r = cliente.post(f"{url}/api/generate", json={"model": "otro", "prompt": "x"})
    assert r.status_code == 404
