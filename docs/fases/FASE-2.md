# Fase 2: vLLM secuencial

**Objetivo:** el mismo `medidor medir` contra un `vllm serve`, con las mismas métricas y el mismo
esquema JSON que Ollama, para poder comparar los dos backends.

**Criterio de aceptación:** el piloto de Qwen3-0.6B en un trozo MIG 1g.5gb reproduce las filas de
**concurrencia 1** de `piloto-mig-0.6b-20261027-155013` dentro de ±5 %. Las filas de concurrencia 4
se validan en la fase 5.

**Requisito:** fase 1 validada (1.2 en verde).

---

## Referencia que hay que reproducir (`referencias/piloto-mig-0.6b/RESUMEN.md`)

`vllm bench serve` de tfgbench, dataset random, `ignore_eos`, 8 solicitudes medidas y 2 de calentamiento por punto:

| entrada | salida | conc. | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | ITL p99 ms | salida tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 1 | 18,9 | 20,2 | 8,3 | 11,0 | 118,5 |
| 1024 | 128 | 1 | 57,6 | 57,8 | 8,9 | 10,3 | 107,6 |

Con TTFT de ~19 ms, un ±5 % es ±1 ms: comparar las **p50**, no las medias, y anotar el loadavg.

## Entregables

### 2.1 `src/medidor/backends/vllm.py` (`BackendVllm`)

Reutilizar la lógica de `referencias/tfgbench/cliente.py` (aiohttp), reescrita con `httpx` como el
backend de Ollama.

- **Generación:** `POST /v1/completions` con `stream: true`,
  `stream_options: {include_usage: true, continuous_usage_stats: true}`, `max_tokens`, `temperature`,
  `seed`, `ignore_eos` (por config) y, si se quiere, `min_tokens = max_tokens`.
- **Parseo SSE:** líneas `data: {...}` hasta `data: [DONE]`. Con `continuous_usage_stats`, cada
  fragmento trae `usage.completion_tokens` acumulado: la diferencia da los **tokens por fragmento**
  (`Fragmento.n_tokens`). Así se cumple "ITL ≠ token" de §2.
- **`metricas_servidor(final)`:** `prompt_tokens`, `completion_tokens` y `finish_reason`, más las tres
  claves comunes `tokens_entrada`, `tokens_salida` y `finish_reason`. vLLM no da duraciones por
  solicitud en la respuesta: `prefill_tok_s` y `decode_tok_s` del servidor quedan vacíos, y en vLLM
  las métricas que valen son las del cliente.
- **`salud()`:** `GET /health` con 200. **`modelos()`:** `GET /v1/models`, que trae `max_model_len`, usado por G6.

### 2.2 Prompts en vLLM: tokenización exacta, sin calibrar

vLLM tiene `POST /tokenize` (`{"model": ..., "prompt": ...}` → `{"count": N, "tokens": [...]}`).
Se usa en lugar de los sondeos de Ollama. `prompts.calibrar` ya acepta cualquier función
`medir_tokens(palabras)`: aquí esa función llama a `/tokenize` y no genera nada. Así el resultado es
exacto, no hace falta descargar el tokenizer al cliente ni añadir dependencias (`transformers`).

Alternativa (la que usaba tfgbench): mandar **ids** de tokens aleatorios en `prompt`. Da una longitud
exacta, pero el prompt deja de ser texto y no se parece a lo que se manda a Ollama. Se recomienda
`/tokenize` con texto, que es comparable entre los dos backends.

### 2.3 Config

```toml
[servidor]
backend = "vllm"
url     = "http://127.0.0.1:8100"        # nunca 8000 (cedia-small en compute-0-0)
modelo  = "Qwen/Qwen3-0.6B"
timeout_s = 300

[servidor.opciones_vllm]                 # nueva subtabla, obligatoria con backend = "vllm"
ignore_eos = true
log        = "~/tfg-jose-ureta/logs_mig/vllm_serve_{job}.log"   # para `medidor log` y G-vLLM
```

`config.py`: `opciones_vllm` solo es válida con `backend = "vllm"` y `opciones_ollama` solo con
`"ollama"`. Mantener la validación estricta de claves.

### 2.4 Guardias para vLLM

| guardia | en vLLM |
|---|---|
| G1 | `GET /health` → 200 |
| G2 | igual que en Ollama (memoria GPU > 0). **En MIG**, `nvidia-smi --query-gpu=memory.used` puede dar `[N/A]`: usar `nvidia-smi -q -d MEMORY` o la memoria del trozo MIG en `nvidia-smi` sin `-L`, y documentar lo que se usó |
| G3 | **no aplica** (es de Ollama). Registrarla como `{"id": "G3", "ok": true, "mensaje": "no aplica a vLLM"}` para que la lista de guardias sea la misma |
| G5 | el modelo está en `/v1/models` |
| G6 | `max_model_len` de `/v1/models` ≥ entrada máx + salida máx |
| G7 | `/tokenize`: los tokens del prompt calibrado coinciden con los pedidos (exacto) |
| G8 | igual |
| **G9 (nueva)** | caché de prefijos: si `/metrics` expone `vllm:prefix_cache_hits` (o `gpu_prefix_cache_hit_rate`), comprobar que no sube durante los sondeos. Si no se puede leer, aviso, no error. `servir_vllm.sh` arranca con `--no-enable-prefix-caching` |

### 2.5 `tanda.py`

- Quitar la restricción "solo Ollama" de `comprobar_fase`.
- Elegir el backend según la config. Mismo orden de guardias, con la "carga del modelo" solo en
  Ollama (vLLM ya tiene el modelo cargado cuando `/health` responde).
- `NOTAS_OLLAMA` → notas según el backend (vLLM: "ITL entre fragmentos SSE; tokens_por_fragmento desde
  usage continuo").

### 2.6 Servidor falso (`tests/servidor_falso.py`)

Ya imita `/health`, `/v1/models` y `/v1/completions` en SSE. Falta añadir:

- `POST /tokenize` con el tokenizador falso (un token por palabra).
- `continuous_usage_stats`: `usage` acumulado en cada fragmento.
- `ignore_eos`: aceptarlo. Opcionalmente, `Ajustes.eos_en` para simular una salida corta cuando no se pide.
- `GET /metrics` con un contador de caché de prefijos configurable.

## Tests

- Las métricas contra latencias fijas, como `test_backend_ollama.py`, pero en SSE. Con
  `tokens_por_fragmento = 3`, `tokens_por_fragmento` sale `[3, 3, …]` y el TPOT se calcula con tokens, no con fragmentos.
- G6 con `max_model_len` pequeño, G7 exacto con `/tokenize`, G9 con el contador de caché subiendo.
- De punta a punta con `backend = "vllm"`: el mismo juego que `test_tanda.py`.
- La config: `opciones_vllm` con backend Ollama es un error, y viceversa.

## Validación en el clúster (tú)

```bash
# Trozo MIG (Qwen3-0.6B cabe en 5 GB; --max-model-len 2048 porque 4096 no cabe en la KV)
salloc -p interactive --gres=gpu:a100_1g.5gb:1 -c 4 --mem=16G -t 01:00:00
nvidia-smi -L                                    # debe decir MIG 1g.5gb
cd ~/tfg-jose-ureta
LOG=logs_mig/vllm_serve_$SLURM_JOB_ID.log
MODEL=Qwen/Qwen3-0.6B MAXLEN=2048 PORT=8100 nohup bash experimentos/servidor/servir_vllm.sh > "$LOG" 2>&1 &
#   (si vllm._C no importa, el script ya corre parche_glibc.sh)
curl -s 127.0.0.1:8100/health && nvidia-smi       # /health 200 Y memoria > 0 antes de medir
cd ~/medidor-a100 && git pull
.venv/bin/python -m medidor comprobar -c configs/piloto-mig-vllm-qwen3-0.6b.toml
.venv/bin/python -m medidor medir     -c configs/piloto-mig-vllm-qwen3-0.6b.toml
```

Ajustar `configs/piloto-mig-vllm-qwen3-0.6b.toml` a la referencia: entradas `[128, 1024]`, salida
`[128]`, 8 repeticiones, 2 de calentamiento y `concurrencias = [1]`.

## Instrucción para Claude Code (PC)

> Repo medidor-a100: lee `DESARROLLO.md`, `docs/fases/README.md` y `docs/fases/FASE-2.md`, y el
> código de la fase 1 (`src/medidor/backends/ollama.py`, `tanda.py`, `guardias.py`). Implementa la fase 2
> (backend vLLM) siguiendo el documento; el cliente de referencia está en `referencias/tfgbench/cliente.py`.
> Antes de escribir código, propón los pasos numerados y espera mi aprobación. No hagas commit.
> Los tests deben pasar en Windows sin red ni GPU, usando `tests/servidor_falso.py`.
