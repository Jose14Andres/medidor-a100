# medidor-a100 — documento de arranque del desarrollo

Herramienta pequeña para conectarse al nodo con la A100 (HPC de CEDIA) y medir el rendimiento de
inferencia de LLM servidos con **vLLM** y con **Ollama**, con salida siempre en **JSON** para las
gráficas de la tesis.

- Estado: fase 0 cerrada (20-nov-2026, 41 tests en verde en compute-0-1); fase 1 en desarrollo.
- Fecha: 19-nov-2026 (diseño); 20-nov-2026 (decisiones §12.1–2 y fase 1).
- Repositorio: <https://github.com/Jose14Andres/medidor-a100>, clonado y desarrollado en el clúster
  (`~/medidor-a100`, nodo de cómputo con `module load python/3.11` o el py311 de miniconda).

---

## 1. Objetivo

Hoy la medición está repartida en piezas sueltas:

| pieza | qué hace | limitación |
|---|---|---|
| `experimentos/src/tfgbench/` | cliente streaming contra vLLM (`/v1/completions`): TTFT, TPOT, ITL | solo vLLM, sin telemetría de GPU |
| `ollama/medir_ollama.py` | prefill/decode contra Ollama (`/api/generate`) | script suelto, sin concurrencia, sin telemetría |
| `chat_stream.py` | chat interactivo con TTFT/TPOT | manual, no produce datos de tesis |
| `resumen_pd.py`, `ollama/resumir.py` | agregan resultados | formatos distintos por backend |

`medidor-a100` unifica todo en **una sola herramienta** que:

1. localiza el nodo y la GPU que realmente tengo reservados (Slurm, MIG o A100 entera);
2. comprueba que el servidor está de verdad listo antes de medir (ver §6, guardias);
3. lanza la misma batería de pruebas contra vLLM u Ollama con un único fichero de configuración;
4. muestrea la GPU durante la medición (memoria, uso, potencia, temperatura, relojes);
5. guarda `entorno.json`, `datos.json` y `resumen.json` con un esquema común y versionado;
6. genera gráficas de calidad de tesis a partir de esos JSON;
7. empaqueta los resultados en un zip en `$HOME` (Jupyter no permite descargar carpetas).

Fuera de alcance (por ahora): arrancar o parar servidores automáticamente, evaluar calidad de
respuesta (exactitud, benchmarks tipo MMLU), interfaz web.

---

## 2. Métricas

Todas en el lado cliente salvo que se indique. Unidades: segundos en los JSON, ms en las tablas.

| métrica | definición | fuente |
|---|---|---|
| **TTFT** | envío de la solicitud → primer fragmento con texto | cliente |
| **E2E** | envío → último fragmento | cliente |
| **TPOT** | `(E2E − TTFT) / (tokens_salida − 1)` | cliente |
| **ITL** | intervalos entre fragmentos consecutivos (lista completa por solicitud) | cliente |
| **prefill tok/s** | `tokens_entrada / TTFT` (cliente) y `prompt_eval_count / prompt_eval_duration` (Ollama) | ambos |
| **decode tok/s** | `(tokens_salida − 1) / (E2E − TTFT)` y `eval_count / eval_duration` (Ollama) | ambos |
| **load_duration** | tiempo de carga del modelo informado por Ollama | servidor |
| **throughput agregado** | tokens de salida totales / duración de la tanda (con concurrencia) | cliente |
| **goodput** (opcional) | % de solicitudes que cumplen un SLO (p. ej. TTFT < 1 s y TPOT < 50 ms) | cliente |
| **GPU** | memoria usada, % uso, potencia (W), temperatura, reloj SM; muestreo cada 500 ms | `nvidia-smi` en el nodo |

Estadísticos en `resumen.json` por punto de medida: n, media, desviación típica, mín, p50, p90,
p95, p99, máx. Las repeticiones de calentamiento se guardan en `datos.json` marcadas con
`"calentamiento": true` y **no** entran en el resumen.

Notas de medición que el código debe respetar:

- **ITL ≠ token**: un fragmento SSE puede traer más de un token. Guardar cuántos tokens trae cada
  fragmento cuando el backend lo informe; si no, documentarlo en el resumen.
- **Tokens reales, no pedidos**: contar los tokens que devuelve el servidor (`usage` en vLLM con
  `stream_options.include_usage`, `eval_count` en Ollama). Si difieren de lo pedido, marcarlo.
- **Longitud de salida fija**: vLLM admite `ignore_eos` / `min_tokens`; Ollama 0.24 no tiene
  `ignore_eos`, así que la salida puede acabar antes. Guardar `finish_reason` / `done_reason`.
- **Sin caché de prefijos**: cada solicitud lleva un prefijo único (como ya hace
  `medir_ollama.py`), o el prefill de las repeticiones sale artificialmente corto.
- **Contexto de Ollama**: fijar `options.num_ctx` ≥ entrada + salida. Si no, Ollama **trunca el
  prompt en silencio**. Comprobar después que `prompt_eval_count` ≈ tokens de entrada esperados.
- **Ollama `keep_alive`**: fijarlo (p. ej. `"30m"`) para que el modelo no se descargue entre
  solicitudes y `load_duration` no contamine el TTFT.

---

## 3. Arquitectura

```
┌──────────── máquina cliente ────────────┐          ┌──────────── nodo GPU (Slurm) ────────────┐
│ medidor (CLI, Python 3.11)               │          │ vLLM  127.0.0.1:8100  (/v1, /health)     │
│  ├─ conexion   (Slurm, túnel SSH)        │  SSH -L  │ Ollama 127.0.0.1:11434 (/api/*)          │
│  ├─ guardias   (preflight)               │ ───────► │   (dentro de enroot jos_env)             │
│  ├─ backends   (vllm | ollama)           │          │ agente de telemetría:                    │
│  ├─ carga      (secuencial/concurrente)  │          │   nvidia-smi --query-gpu ... -lms 500    │
│  ├─ telemetria (lee CSV del nodo)        │          │   → /home/.../telemetria_<job>.csv       │
│  ├─ esquema / resumen / graficas         │          └──────────────────────────────────────────┘
│  └─ exportar (zip en $HOME)              │
└──────────────────────────────────────────┘
```

### 3.1 Dónde corre el cliente

| modo | cuándo | ventaja | coste |
|---|---|---|---|
| **A. en el propio nodo** (recomendado para datos de tesis) | mediciones que van a gráficas | TTFT sin latencia de red | requiere Python 3.11 en el host (`module load python/3.11`); `jos_env` no tiene python3 |
| **B. remoto por túnel SSH** | pruebas interactivas desde el portátil | cómodo | la latencia de red entra en el TTFT; queda registrada en `entorno.json` |

El modo se guarda siempre en `entorno.json` (`"modo_cliente": "nodo" | "tunel"`) y en modo B se
mide el RTT con 20 pings HTTP a `/health` antes de empezar.

### 3.2 Conexión

- Nodo reservado: `squeue -u $USER -h -o "%i %N %P %T"`.
- Túnel (modo B): `ssh -N -L 8100:127.0.0.1:8100 -L 11434:127.0.0.1:11434 -J andres.rubio@login1 andres.rubio@<nodo>`.
- GPU real: `nvidia-smi -L` (el `nvidia-smi` sin `-L` muestra la A100 padre aunque tenga un trozo MIG de 5 GB).
- En compute-0-0, `localhost:8000` lo ocupa un servicio ajeno (`cedia-small`): **nunca** usar el 8000 por defecto.

### 3.3 Interfaz común de backend

```python
class Backend(Protocol):
    nombre: str                                   # "vllm" | "ollama"
    def salud(self) -> EstadoSalud: ...           # /health (vLLM) o /api/version + /api/ps (Ollama)
    def modelos(self) -> list[str]: ...           # /v1/models o /api/tags
    def info_modelo(self, m: str) -> dict: ...    # cuantización, tamaño, contexto
    async def generar(self, sol: Solicitud) -> AsyncIterator[Fragmento]: ...
    def metricas_servidor(self, final: dict) -> dict: ...  # prompt_eval_*, eval_*, load_duration, usage
```

`Fragmento` = `(t_monotonic, texto, n_tokens | None)`. Toda la temporización usa
`time.perf_counter()`; las marcas de reloj de pared (`datetime.now(UTC)`) solo sirven para alinear
con la telemetría de GPU.

---

## 4. Configuración (TOML)

Un fichero por experimento, en `configs/`. Ejemplo:

```toml
[experimento]
nombre      = "e1-prefill-ollama-nemotron"
descripcion = "Barrido de longitud de entrada, salida fija, concurrencia 1"

[servidor]
backend = "ollama"                 # "vllm" | "ollama"
url     = "http://127.0.0.1:11434"
modelo  = "nemotron-3-nano:30b"

[servidor.opciones_ollama]
num_ctx    = 8192
keep_alive = "30m"

[carga]
entradas      = [128, 512, 2048, 4096]   # tokens de entrada
salidas       = [128]                    # tokens de salida
concurrencias = [1]
repeticiones  = 10
calentamiento = 2
modo          = "cerrado"                # "cerrado" (N concurrentes fijos) | "poisson" (tasa)
# tasa_rps    = 2.0                      # solo con modo = "poisson"
semilla       = 1234
temperatura   = 0.0

[telemetria]
activar      = true
intervalo_ms = 500

[salida]
dir = "resultados"                 # se crea resultados/<nombre>-<AAAAMMDD-HHMMSS>/
```

Validación estricta al cargar: claves desconocidas = error (evita erratas que invalidan una tanda).

---

## 5. Salida JSON (esquema v1)

```
resultados/<nombre>-<fecha>/
├── config.toml        copia exacta de la configuración usada
├── entorno.json       nodo, job, GPU (nvidia-smi -L), driver, versiones, modo cliente, RTT, loadavg
├── datos.json         una entrada por solicitud (incluidas las de calentamiento)
├── resumen.json       agregados por punto (entrada, salida, concurrencia)
├── telemetria.json    muestras de GPU con marca de tiempo
├── log.txt            log del cliente
└── graficas/          PNG + PDF (vectorial, para LaTeX)
```

Cada JSON lleva `"esquema": "medidor-a100/v1"`. Ejemplo de una entrada de `datos.json`:

```json
{
  "id": 17, "punto": {"entrada": 512, "salida": 128, "concurrencia": 1},
  "calentamiento": false, "backend": "ollama", "modelo": "nemotron-3-nano:30b",
  "t_inicio_utc": "2026-11-19T10:21:03.412Z",
  "ttft_s": 0.184, "e2e_s": 2.913, "tpot_s": 0.02149, "itl_s": [0.021, 0.022, "..."],
  "tokens_entrada": 512, "tokens_salida": 128, "finish_reason": "length",
  "servidor": {"prompt_eval_count": 512, "prompt_eval_duration_s": 0.151,
               "eval_count": 128, "eval_duration_s": 2.70, "load_duration_s": 0.012},
  "error": null
}
```

Compatibilidad: un exportador `medidor exportar-vllm-bench` escribe además el formato de
`vllm bench serve --save-result`, para que `resumen_pd.py` siga funcionando con los datos nuevos.

---

## 6. Guardias (preflight) — obligatorias antes de medir

Reglas del TFG llevadas al código. Si una guardia falla, el programa **para y explica por qué**;
no reintenta en bucle ni improvisa otra cosa.

| # | comprobación | si falla |
|---|---|---|
| G1 | `/health` (vLLM) o `/api/version` (Ollama) responde 200 | "servidor no listo" |
| G2 | memoria GPU usada > 0 en `nvidia-smi` | "el servidor responde pero no está en GPU" |
| G3 | Ollama: el log contiene `inference compute` con `CUDA` y `/api/ps` da `size_vram` > 0 | "Ollama cayó a CPU" (pasa cuando el descubrimiento de GPU supera sus 30 s) |
| G4 | `nvidia-smi -L` coincide con la GPU esperada en la config (`gpu_esperada = "A100-40GB"` o `"MIG 1g.5gb"`) | aviso o error según config |
| G5 | el modelo pedido aparece en `/v1/models` o `/api/tags` | error |
| G6 | `num_ctx` (Ollama) / `max_model_len` (vLLM) ≥ entrada máx + salida máx | error |
| G7 | solicitud de sondeo: `prompt_eval_count` ≈ tokens de entrada pedidos (±2 %) | error: truncado o tokenizador distinto |
| G8 | directorio de salida en `/home` (no en `/scratch`, que se borra al acabar el job) | error |

Reglas de comportamiento del programa:

- **No mata nada.** Nunca hace `kill`, `scancel` ni `pkill`. Si hay que parar un servidor, lo hace la
  persona, después de leer las últimas 40 líneas de su log (el programa las imprime con `medidor log`).
- **Los tiempos de espera no se suben para "arreglar".** Los tiempos de espera son parámetros
  explícitos y se registran en `entorno.json`. Si salta uno, se informa de dónde estaba la solicitud.
- **No sobrescribe datos.** Cada tanda escribe en una carpeta nueva con marca de tiempo; nunca toca
  `tesis/resultados/`, `resultados/` ni `ollama/resultados/` existentes.
- **Solo números medidos.** El resumen no extrapola ni rellena puntos que faltan; un punto con
  errores muestra `n` real y el número de errores.

---

## 7. CLI

```
medidor entorno                      # versiones, Python, config válida (sin red)
medidor nodo                         # job, nodo, GPU real (squeue + nvidia-smi -L)
medidor comprobar  -c configs/x.toml # guardias G1–G8 sin medir
medidor medir      -c configs/x.toml # guardias + tanda completa + resumen
medidor chat       -c configs/x.toml # chat interactivo con TTFT/TPOT (sustituye a chat_stream.py)
medidor log        --servidor ollama --lineas 40
medidor resumir    resultados/<dir>  # regenera resumen.json desde datos.json
medidor graficas   resultados/<dir> [<dir> ...]   # una tanda o comparativa entre tandas
medidor exportar                     # actualiza resumen global + zip en $HOME
```

Códigos de salida: 0 bien, 1 fallo de guardia, 2 error de configuración, 3 tanda con errores parciales.

---

## 8. Gráficas previstas

Generadas con matplotlib, estilo único (tipografía serif, tamaño de columna de LaTeX, PDF vectorial):

1. TTFT vs. tokens de entrada (p50 con banda p10–p90), una línea por backend/modelo.
2. Prefill tok/s vs. tokens de entrada.
3. TPOT / decode tok/s vs. concurrencia.
4. Throughput agregado vs. concurrencia (curva de saturación).
5. CDF de ITL por punto.
6. Serie temporal de la GPU (memoria, potencia, uso) superpuesta a las fases de la tanda.
7. Comparativa vLLM vs. Ollama con el mismo modelo y el mismo punto.

---

## 9. Estructura del proyecto

```
medidor-a100/
├── DESARROLLO.md            este documento
├── README.md                uso (cuando haya código)
├── pyproject.toml           Python ≥ 3.11; dependencias: httpx, matplotlib, numpy
├── requirements.lock        con hashes (mismo criterio que experimentos/)
├── configs/                 un TOML por experimento
├── src/medidor/
│   ├── __main__.py · cli.py
│   ├── config.py            carga y validación del TOML (tomllib)
│   ├── conexion.py          squeue, nvidia-smi -L, túnel, RTT
│   ├── guardias.py          G1–G8
│   ├── backends/{base,vllm,ollama}.py
│   ├── prompts.py           generación de prompts de N tokens con prefijo único
│   ├── carga.py             planificador: secuencial, concurrencia cerrada, Poisson
│   ├── telemetria.py        lanza/lee el muestreo de nvidia-smi
│   ├── esquema.py           dataclasses + escritura JSON versionada
│   ├── resumen.py · graficas.py · exportar.py
└── tests/
    ├── servidor_falso.py    servidor SSE local que imita vLLM y Ollama con latencias fijas
    └── test_*.py            pytest; no necesitan GPU
```

Reutilizar en lugar de reescribir: `tfgbench/cliente.py` (streaming OpenAI), `tfgbench/tokens.py`
(tokenizer local de `hf-cache`), `tfgbench/esquema.py` y la lógica de `ollama/medir_ollama.py`.

---

## 10. Plan por fases

Cada fase termina con algo ejecutable y verificado. Se aprueban una a una.

| fase | entregable | criterio de aceptación |
|---|---|---|
| **0. Esqueleto** | `pyproject`, venv, `medidor entorno`, servidor falso, CI local con pytest | `pytest` verde sin red ni GPU |
| **1. Ollama secuencial** | backend Ollama + guardias G1–G3, G5–G8 + `datos.json`/`resumen.json` | con nemotron-3-nano:30b, los números de prefill/decode coinciden (±5 %) con la tanda del job 36268 de `medir_ollama.py` |
| **2. vLLM secuencial** | backend vLLM reutilizando tfgbench | piloto Qwen3-0.6B en MIG reproduce las filas c=1 de `piloto-mig-0.6b-20261027-155013` (±5 %) |
| **3. Telemetría GPU** | muestreo `nvidia-smi` alineado por marcas UTC | `telemetria.json` cubre toda la tanda; memoria > 0 durante las solicitudes |
| **4. Vista en tiempo real** | `medidor panel` (web por túnel SSH, solo lee ficheros); opcional `medidor proxy` | solicitudes en el panel ≤ 2 s; con panel abierto TTFT p50 ±2 % |
| **5. Concurrencia** | modos cerrado y Poisson, throughput agregado | con el servidor falso, throughput medido = teórico (±2 %); filas c=4 del piloto (±5 %) |
| **6. Gráficas y exportación** | `graficas`, `exportar` (zip en `$HOME`) | las 7 gráficas salen en PNG y PDF desde una tanda real |
| **7. Chat y comparativas** | `medidor chat`, gráficas entre tandas | sustituye a `chat_stream.py` |

El 20-nov-2026 la vista en tiempo real (antes no estaba en el plan) pasó a la fase 4, por delante de
la concurrencia: es el objetivo de uso de la herramienta. El detalle de cada fase está en
[`docs/fases/`](docs/fases/README.md).

Las fases 1 y 2 validan la herramienta nueva contra datos ya medidos: si no coinciden, se para y se
investiga antes de seguir.

---

## 11. Entorno del clúster (hechos medidos que condicionan el diseño)

- Driver NVIDIA 535 → Ollama fijado en **0.24.0** (`~/ollama`, con cuda-compat 12.8). No actualizar.
- Ollama se arranca dentro de enroot `jos_env` (~45 s). El cliente corre en el host porque `jos_env` no trae python3.
- Detener Ollama: `pkill -x ollama`, **nunca** `pkill -f` (mata la propia shell). Lo hace la persona, no el programa.
- Modelos Ollama disponibles: `bge-m3`, `qwen3-coder:30b`, `nemotron-3-nano:30b`, `nemotron-3-super` (80,9 GiB: no cabe en 40 GB).
- vLLM 0.20 en `venv-vllm` necesita `parche_glibc.sh` tras cada reinstalación.
- QoS: un trabajo a la vez por usuario (un `salloc` abierto bloquea el `sbatch`).
- `/scratch/$USER/<jobid>/` se borra al terminar el job: los resultados van siempre a `/home`.
- NFS de `/home` lento con loadavg alto: registrar `loadavg` del nodo en `entorno.json` al empezar y al acabar.

---

## 12. Decisiones

Tomadas el 20-nov-2026:

1. **Prompts para Ollama → calibración por sondeo.** No hay tokenizer de nemotron en `hf-cache` y
   habría que bajar uno por modelo. `prompts.calibrar` lanza sondeos (`num_predict = 1`), lee
   `prompt_eval_count` y ajusta el número de palabras (modelo lineal tokens ≈ a + b·palabras, como
   mucho 8 sondeos) hasta quedar a ≤ 1 token del objetivo y dentro de ±2 %. G7 falla si no lo consigue.
   Cada solicitud guarda `tokens_entrada` (reales) y `tokens_entrada_pedidos`; si se salen de ±2 %, se
   marca `entrada_fuera_tolerancia` y el resumen lo cuenta.
2. **Plantilla → `raw` para la tesis** (`/api/generate` con `raw: true`, comparable con
   `/v1/completions` de vLLM y con el job 36268). La plantilla de chat queda para `medidor chat`.
3. **Validación de la fase 1 → config propia** `configs/v1-validacion-36268.toml`: entradas
   170/615/2390 tokens (lo que dieron las 128/512/2048 palabras del 36268), salida 128, 5 repeticiones
   y 1 de calentamiento, `num_ctx` 8192.

Detalles de implementación de la fase 1 que fijan comportamiento:

- Orden de guardias: G8 → G6 → G1 → G5 → carga del modelo → G3 → G2 → G7. La carga ("Hola",
  1 token) se cronometra aparte y queda en `entorno.json` (`carga_modelo`).
- G3 lee el log de `ollama serve` desde su último `Listening on`: exige `inference compute` con
  `library=CUDA` y `offloaded N/N layers`, y en `/api/ps` `size_vram > 0` e igual a `size`. Ruta en
  `servidor.opciones_ollama.log`, con `{job}` = `SLURM_JOB_ID`.
- G8 exige que la salida esté bajo `/home` (`guardias.RAICES_PERMITIDAS`).
- `servidor.timeout_s` (900 s por defecto) es el tiempo máximo **sin recibir datos**, no el total;
  incluye la carga del modelo (181 s el 16-nov). Va en `entorno.json` junto a los demás tiempos de espera.
- Tras 3 solicitudes seguidas con error, la tanda para sin reintentar (código 3). Si no, un servidor
  colgado costaría 900 s por cada solicitud que queda.
- `datos.jsonl` se escribe solicitud a solicitud; `datos.json`, `resumen.json` y `entorno.json` se
  escriben al final, también si falla una guardia o se interrumpe la tanda (con `estado`).
- Las funciones de consulta de la interfaz `Backend` son `async` (§3.3), porque la fase 4
  (concurrencia) reutiliza el mismo cliente.

Abiertas:

4. **Telemetría en MIG**: en un trozo MIG, `nvidia-smi` no da % de uso por instancia. ¿Basta con
   la memoria, o se intenta DCGM (si está instalado en el nodo)?
5. **Python del cliente**: 3.11 (como tfgbench) frente a solo biblioteca estándar 3.8 (como
   `medir_ollama.py`, que corre en cualquier sitio). De hecho se usa 3.11 con dependencias mínimas.
6. **Interfaz**: solo CLI, o añadir más adelante una TUI (p. ej. `textual`) para ver la tanda en vivo.
