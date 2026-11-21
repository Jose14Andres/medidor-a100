# Fase 1: Ollama secuencial

**Estado (20-nov-2026):** código terminado, con 81 tests en verde en compute-0-1 y en Windows. El
comparador (1.1) está hecho en el PC (95 tests en verde). **Falta**: la validación real en la A100
(1.2, en el clúster).

**Criterio de aceptación (DESARROLLO.md §10):** con nemotron-3-nano:30b, prefill y decode tok/s del
servidor dentro de **±5 %** de la tanda del job 36268 de `medir_ollama.py`, punto a punto.

---

## Qué está hecho

| fichero | qué hace |
|---|---|
| `src/medidor/backends/base.py` | interfaz `Backend` (async), `Solicitud`, `Fragmento`, `Resultado`; `medir()` calcula TTFT, E2E, TPOT e ITL con `perf_counter` y nunca lanza: deja el error con el sitio donde estaba la solicitud |
| `src/medidor/backends/ollama.py` | `/api/generate` en streaming NDJSON con `raw: true`, `num_ctx`, `keep_alive`, `num_predict`, `temperature`, `seed`; `/api/version`, `/api/tags`, `/api/ps`, `/api/show` |
| `src/medidor/prompts.py` | prefijo único `Solicitud <id>-<nonce>.` (sin caché KV) y calibración por sondeo con `prompt_eval_count` |
| `src/medidor/guardias.py` | G1, G2, G3, G5, G6, G7 y G8 |
| `src/medidor/conexion.py` | `nvidia-smi` (memoria, `-L`), loadavg, job de Slurm |
| `src/medidor/resumen.py` | n, media, desv, mín, p50/p90/p95/p99, máx por punto; cuenta errores, salidas cortas y entradas fuera de tolerancia |
| `src/medidor/tanda.py` | orden de guardias, carga del modelo, calibración, tanda secuencial, escritura de resultados |
| `src/medidor/cli.py` | `entorno`, `comprobar`, `medir`, `resumir`, `comparar` |
| `src/medidor/comparar.py` | tarea 1.1: compara con el `resumen.json` de `medir_ollama.py` (ver abajo) |
| `configs/v1-validacion-36268.toml` | la tanda de validación |
| `tests/test_{prompts,backend_ollama,guardias,resumen,tanda}.py` | 81 tests en total |

### Decisiones tomadas (DESARROLLO.md §12)

- **Calibración por sondeo:** hasta 8 sondeos con `num_predict = 1`; para cuando el error es ≤ 1 token y
  cabe en ±2 %. No hay tokenizer de nemotron en `hf-cache`.
- **`raw`** para la tesis: es comparable con `/v1/completions` de vLLM y con el 36268.
- **Entradas en tokens:** 170/615/2390, que es lo que dieron las 128/512/2048 palabras del 36268.

### Comportamiento que fija la fase 1

- Orden: **G8 → G6 → G1 → G5 → carga del modelo → G3 → G2 → G7 → tanda**.
- **G3** lee el log de `ollama serve` desde su último `Listening on` y exige `library=CUDA` y
  `offloaded N/N layers`. En `/api/ps` exige `size_vram > 0` e igual a `size`. Se probó con extractos
  reales: job 36268 (GPU, 53/53) y job 36509 (caído a CPU, 0/49).
- **`servidor.timeout_s`** (900 s) es el tiempo **sin recibir datos**, no el total. Incluye la carga
  del modelo (181 s el 16-nov).
- **3 errores seguidos** paran la tanda con código 3, sin reintentar.
- **Cada tanda** crea `resultados/<nombre>-<AAAAMMDD-HHMMSS>/` con `config.toml`, `log.txt`,
  `datos.jsonl` (se escribe solicitud a solicitud), `datos.json`, `resumen.json` y `entorno.json`
  (estos tres al final, también si falla una guardia o se interrumpe).
- **Códigos de salida:** 0 bien, 1 guardia, 2 config, 3 tanda con errores.

---

## 1.1 Hecho en el PC: comparador con la referencia

Hecho el 20-nov-2026 (`src/medidor/comparar.py`, `tests/test_comparar.py`). Decisiones: el
emparejamiento es uno a uno (dos puntos de la referencia no comparten pareja); un punto de la
referencia sin pareja o una métrica sin valor es NO PASA; ±2 % en tokens solo avisa; el borde de ±5 %
es inclusivo. Salida: 0 pasa, 1 no pasa, 2 entrada ilegible o con otro formato.

Hace falta una herramienta que diga "pasa / no pasa" sin hacer cuentas a mano.

**Entregable:** orden `medidor comparar <dir_tanda> --referencia <resumen.json de medir_ollama.py>`
(en `src/medidor/comparar.py`) que:

1. Lee `resumen.json` del medidor (`puntos[].punto.entrada`, `servidor_prefill_tok_s.p50`,
   `servidor_decode_tok_s.p50`, `tokens_entrada.media`).
2. Lee el `resumen.json` de `medir_ollama.py` (formato distinto: `puntos[].entrada_palabras`,
   `prompt_tokens.media`, `servidor_prefill_tok_s.mediana`, `servidor_decode_tok_s.mediana`).
3. Empareja los puntos por **tokens reales** (el más cercano a `prompt_tokens.media`) y avisa si la
   diferencia de tokens supera el 2 %.
4. Imprime una tabla (punto, referencia, nuevo, diferencia %, PASA/NO PASA con ±5 %) y escribe
   `comparacion.json` en la carpeta de la tanda (sin sobrescribir: `comparacion-<fecha>.json` si ya existe).
5. Sale con 0 si todo pasa y con 1 si algo no pasa.

Además informa de TTFT y TPOT del cliente, **sin entrar en el criterio**. El 36268 tuvo un TTFT de
cliente de ~0,98 s frente a 0,11 s de prefill del servidor en el punto de 128 palabras. Esa
diferencia hay que registrarla, no explicarla todavía.

**Referencia** (`referencias/resumen_36268.json`, medianas):

| palabras | tokens reales | prefill tok/s | decode tok/s | TTFT cliente s |
|---:|---:|---:|---:|---:|
| 128 | 170 | 1513,6 | 152,0 | 0,983 |
| 512 | 614,8 | 2204,8 | 153,4 | 1,599 |
| 2048 | 2389,8 | 3007,2 | 152,1 | 2,554 |

**Tests:** fixture con un `resumen.json` falso de cada formato; casos que pasan, que no pasan (±5 %
justo en el borde), puntos sin pareja y la escritura sin sobrescribir.

---

## 1.2 Pendiente en el clúster: validación en la A100 (tú)

Necesita una **A100 entera** (no un trozo MIG: nemotron-3-nano ocupa ~24,8 GB).

```bash
# 1. Reserva (QoS: un trabajo a la vez; cierra cualquier otro salloc antes)
salloc -p gpu --gres=gpu:a100-sxm4-40gb:1 -c 8 --mem=64G -t 02:00:00
nvidia-smi -L                                   # debe decir A100-SXM4-40GB, sin MIG

# 2. Ollama dentro de jos_env, con el log donde lo busca G3 (~45 s de enroot + carga)
LOG=~/tfg-jose-ureta/ollama/logs/ollama_serve_$SLURM_JOB_ID.log
nohup enroot start jos_env bash -c 'export LD_LIBRARY_PATH=$HOME/ollama/cudacompat/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH \
  OLLAMA_MODELS=$HOME/ollama_jm/models OLLAMA_HOST=127.0.0.1:11434 OLLAMA_KEEP_ALIVE=30m; \
  exec $HOME/ollama/bin/ollama serve' > "$LOG" 2>&1 &
grep "inference compute" "$LOG"                 # repetir hasta que salga; debe decir library=CUDA
#   si dice library=cpu o "failure during GPU discovery": PARAR. Pega las últimas 40 líneas del log.

# 3. Medidor (Python 3.11 del clúster)
cd ~/medidor-a100 && git pull
[ -x .venv/bin/python ] || PYTHON=/scratch/sw/apps/miniconda3/envs/py311/bin/python3.11 ./ci.sh
.venv/bin/python -m medidor comprobar -c configs/v1-validacion-36268.toml   # todas las guardias OK
.venv/bin/python -m medidor medir     -c configs/v1-validacion-36268.toml   # ~5 min + carga del modelo (~3 min)

# 4. Comparación (cuando exista 1.1)
.venv/bin/python -m medidor comparar resultados/v1-validacion-36268-<fecha> \
  --referencia ~/tfg-jose-ureta/ollama/resultados/nemotron-3-nano_30b-20261116-185849/resumen.json
```

La A100 de compute-0-1 se comparte con otros jobs y el loadavg suele estar entre 80 y 160. Queda
registrado en `entorno.json` (`loadavg_inicio` y `loadavg_fin`). El 36268 tuvo un loadavg de ~86.

**Si algo falla, se para y se reporta.** No hay que subir tiempos de espera ni reintentar.

**Si no pasa el ±5 %:** antes de tocar nada, comparar `tokens_entrada` (¿mismos tokens?),
`n_salida_corta` (¿terminó antes por EOS?), `loadavg` y `servidor_load_duration_s` (¿se descargó el modelo?).

---

## Instrucción para Claude Code (PC)

> Estoy en el repo medidor-a100 (lee `DESARROLLO.md` y `docs/fases/README.md` y `docs/fases/FASE-1.md`).
> La fase 1 está implementada; falta la tarea 1.1: la orden `medidor comparar` contra el `resumen.json`
> de `medir_ollama.py` (formato en `referencias/resumen_36268.json`). Propón los pasos numerados antes
> de escribir código; no hagas commit. Respeta las reglas del README de fases; los tests deben pasar en
> Windows sin red ni GPU.
