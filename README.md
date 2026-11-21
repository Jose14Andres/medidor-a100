# medidor-a100

Mide el rendimiento de inferencia de LLM servidos con vLLM y Ollama en la A100 de CEDIA, con salida
en JSON. El diseño completo está en [DESARROLLO.md](DESARROLLO.md) y el plan por fases en
[docs/fases/](docs/fases/README.md).

Estado: **fase 1 (Ollama secuencial)**: `entorno`, `comprobar`, `medir` y `resumir`. Concurrencia,
vLLM, telemetría y gráficas llegan en las fases siguientes (§10). Se desarrolla y se ejecuta en el
clúster de CEDIA.

## Instalación

```bash
module load python/3.11            # en el clúster (o /scratch/sw/apps/miniconda3/envs/py311/bin/python3.11)
python3 -m venv .venv
.venv/bin/pip install --require-hashes -r requirements.lock
.venv/bin/pip install --no-deps -e .
```

`requirements.lock` se regenera con
`pip-compile --generate-hashes --allow-unsafe --strip-extras --extra dev -c restricciones-py311.txt -o requirements.lock pyproject.toml`
(las restricciones evitan versiones que exigen Python ≥ 3.12 si se compila con un Python más nuevo).

## Uso

```bash
medidor entorno                                        # versiones y dependencias (sin red)
medidor entorno -c configs/v1-validacion-36268.toml    # además valida la config
medidor comprobar -c configs/v1-validacion-36268.toml  # guardias G1–G3, G5–G8 (con sondeos), sin medir
medidor medir     -c configs/v1-validacion-36268.toml  # guardias + tanda + resumen
medidor resumir   resultados/<nombre>-<fecha>          # regenera el resumen (sin sobrescribir)
```

Se ejecuta **en el nodo GPU** (modo A, §3.1): las guardias G2 y G3 leen `nvidia-smi` y el log de
`ollama serve`. Antes, arrancar Ollama con su log en la ruta de `servidor.opciones_ollama.log`
(admite `{job}`). El medidor no arranca ni para servidores.

Cada `medir` crea `resultados/<nombre>-<AAAAMMDD-HHMMSS>/` con `config.toml`, `log.txt`,
`datos.jsonl` (solicitud a solicitud), `datos.json`, `resumen.json` y `entorno.json`. Nunca sobrescribe.

Códigos de salida: 0 bien, 1 fallo de guardia, 2 error de configuración, 3 tanda con errores parciales.

## Tests

```bash
./ci.sh            # crea el venv si no existe y ejecuta pytest
```

Los tests no necesitan red ni GPU: usan `tests/servidor_falso.py`, que imita vLLM (SSE) y Ollama
(NDJSON) en 127.0.0.1 con latencias fijas, y `tests/datos/` con extractos reales del log de Ollama
(job 36268 en GPU, job 36509 caído a CPU). También se puede arrancar a mano:
`python tests/servidor_falso.py --puerto 8100`.
