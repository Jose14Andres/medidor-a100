# medidor-a100

Mide el rendimiento de inferencia de LLM servidos con vLLM y Ollama en la A100 de CEDIA, con salida
en JSON. El diseño completo está en [DESARROLLO.md](DESARROLLO.md).

Estado: **fase 0 (esqueleto)**. Solo existe `medidor entorno`; el resto de órdenes llega en las
fases siguientes (§10).

## Instalación

```bash
module load python/3.11            # en el clúster
python3 -m venv .venv
.venv/bin/pip install --require-hashes -r requirements.lock
.venv/bin/pip install --no-deps -e .
```

`requirements.lock` se regenera con
`pip-compile --generate-hashes --allow-unsafe --strip-extras --extra dev -c restricciones-py311.txt -o requirements.lock pyproject.toml`
(las restricciones evitan versiones que exigen Python ≥ 3.12 si se compila con un Python más nuevo).

## Uso

```bash
medidor entorno                                        # versiones y dependencias
medidor entorno -c configs/e1-prefill-ollama-nemotron.toml   # además valida la config
medidor entorno --json
```

Códigos de salida: 0 bien, 1 fallo de guardia, 2 error de configuración, 3 tanda con errores parciales.

## Tests

```bash
./ci.sh            # crea el venv si no existe y ejecuta pytest
```

Los tests no necesitan red ni GPU: usan `tests/servidor_falso.py`, que imita vLLM (SSE) y Ollama
(NDJSON) en 127.0.0.1 con latencias fijas. También se puede arrancar a mano:
`python tests/servidor_falso.py --puerto 8100`.
