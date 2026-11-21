# Fase 4: vista en tiempo real

**Objetivo (el de toda la herramienta):** con el medidor corriendo en el nodo de la A100, abrir en el
navegador del PC un panel que muestre **en vivo** cada solicitud medida (TTFT, TPOT, tokens/s), el
progreso de la tanda, las guardias y la GPU (memoria, potencia, uso), con Ollama o con vLLM.

**Criterio de aceptación:**
1. Durante una tanda real, cada solicitud aparece en el panel del PC **≤ 2 s** después de que se escribe en `datos.jsonl`.
2. **El panel no perturba la medida:** la misma tanda con el panel abierto y cerrado da una TTFT p50 dentro de ±2 %.
3. Al cerrar el panel o perder el túnel, la tanda sigue sin enterarse.

**Requisito:** fases 1–3 (el panel de GPU necesita la telemetría de la 3).

---

## Decisión que hay que tomar antes de empezar (DESARROLLO.md §12, decisión abierta 6)

| opción | cómo se ve | pros | contras |
|---|---|---|---|
| **A. Panel web por túnel SSH (recomendada)** | navegador del PC → `http://localhost:8150` | gráficas en vivo, varias pestañas, nada que instalar en el PC | hace falta un túnel `ssh -L` (el mismo que con Open WebUI en el 8090) |
| B. Terminal (TUI con `rich`/`textual`) | en la misma sesión SSH | sin túnel ni puertos | pocas gráficas; nueva dependencia en el clúster |

Recomendación: **A**, con una vista de texto mínima (`medidor vivo`) como respaldo para cuando solo haya SSH.

**Extra opcional (4.4): proxy de medición.** Para "ver en tiempo real lo que hago con Ollama", también
cuando se chatea desde Open WebUI y no desde una tanda, el medidor puede ponerse **en medio** entre
Open WebUI y Ollama, medir cada solicitud que pasa y mostrarla en el mismo panel. Decide si entra en
esta fase o se deja para después.

---

## Diseño (opción A)

### 4.1 Arquitectura: el panel **solo lee ficheros**

```
medidor medir ──escribe──► resultados/<tanda>/datos.jsonl, telemetria.jsonl, log.txt, entorno.json
                                       │ (lee cada 0,5 s, en otro proceso)
medidor panel ──HTTP 127.0.0.1:8150──► navegador (PC, por túnel SSH)
```

- **El panel es otro proceso** (`medidor panel`), que se lanza en la misma reserva y lee los ficheros
  de la tanda. **No comparte nada con `medir`**: si el panel se cae o se cierra, la medición no se
  entera. Así se cumple el criterio 3, y el 2 porque solo lee ficheros cada 0,5 s.
- **Sin dependencias nuevas en el servidor:** `http.server` (`ThreadingHTTPServer`), como el servidor
  falso, y **Server-Sent Events** (`text/event-stream`) para empujar los datos nuevos al navegador.
- **Página única autocontenida** (HTML + JS + CSS en un fichero del paquete): gráficas en
  `<canvas>`/SVG propio, sin CDN, porque el nodo no tiene por qué tener salida a internet y así la
  página funciona igual por túnel.
- **Escucha solo en `127.0.0.1`.** En un nodo compartido, cualquier usuario del nodo podría verlo:
  los datos no son sensibles, pero el puerto es configurable y se avisa al arrancar.

### 4.2 Endpoints

| ruta | qué devuelve |
|---|---|
| `GET /` | la página del panel |
| `GET /api/tandas` | carpetas de `resultados/` con su estado (de `entorno.json` si existe; si no, "en curso"), la más reciente primero |
| `GET /api/tanda/<nombre>` | `entorno.json` (si existe), config, guardias y lo que haya de `datos.jsonl` y `telemetria.jsonl` |
| `GET /api/eventos?tanda=<nombre>` | **SSE**: un evento `solicitud` por cada línea nueva de `datos.jsonl`, `gpu` por cada línea de `telemetria.jsonl`, `log` por cada línea de `log.txt` y `estado` cuando aparece `entorno.json` (fin de tanda) |

El panel sigue los ficheros como `tail -f`: guarda el offset de cada uno y lee lo nuevo cada 0,5 s.
Tiene que aguantar líneas a medio escribir (sin `\n` final): se esperan a la siguiente lectura.

### 4.3 Qué muestra la página

1. **Cabecera:** experimento, backend y modelo, nodo y job, estado, progreso `n/total` con barra.
   El total sale de la config: puntos × (calentamiento + repeticiones).
2. **Guardias:** G1–G10 en verde o rojo, con su mensaje.
3. **Solicitudes en vivo:** TTFT y TPOT por solicitud (puntos coloreados por punto de medida;
   calentamiento en gris), más prefill y decode tok/s.
4. **Tabla por punto**, que se actualiza sola: n, p50 y p90 de TTFT/TPOT y tok/s del servidor
   (mismas definiciones que `resumen.py`; reutilizar `resumen.estadisticos` para no tener dos cálculos).
5. **GPU:** memoria, potencia y % de uso en el tiempo, con marcas de inicio y fin de cada punto.
6. **Log:** las últimas 40 líneas de `log.txt`.

### 4.4 (Opcional) proxy de medición: `medidor proxy`

```
Open WebUI ──► medidor proxy (127.0.0.1:11435) ──► Ollama (127.0.0.1:11434)
                    └─ por cada /api/generate o /api/chat: TTFT, TPOT y métricas del final → vivo/<fecha>/datos.jsonl
```

- Reenvía **byte a byte** el stream: no lo modifica ni lo bufferiza. Mide con `perf_counter` en el
  propio proxy, así que la latencia que añade el salto local (≈ décimas de ms) queda en el TTFT y se documenta.
- Las solicitudes que no son de generación (`/api/tags`, `/api/ps`, …) se reenvían sin medir.
- Escribe en el mismo formato que una tanda (`datos.jsonl`), con `punto` = tokens reales, así que
  el panel lo muestra sin cambios.
- Open WebUI se apunta al proxy cambiando `OLLAMA_BASE_URL` en `~/openwebui/start_openwebui.sh` a
  `http://127.0.0.1:11435`. Es un cambio que haces tú; el medidor no toca ese script.
- Para vLLM es lo mismo con `/v1/completions` y `/v1/chat/completions`.

### 4.5 Respaldo en texto: `medidor vivo <dir>`

Sigue los mismos ficheros y reimprime cada segundo el progreso y la tabla por punto. Sirve en
cualquier terminal, sin túnel.

## Cómo se usa (cuando esté)

```bash
# en el nodo, dentro de la reserva (dos procesos)
.venv/bin/python -m medidor panel --puerto 8150 &          # avisa: "panel en http://127.0.0.1:8150"
.venv/bin/python -m medidor medir -c configs/v1-validacion-36268.toml
```

```powershell
# en el PC (el mismo esquema que el túnel de Open WebUI)
ssh -N -L 8150:127.0.0.1:8150 -J andres.rubio@hpc.cedia.edu.ec andres.rubio@compute-0-1
# y abrir http://localhost:8150 en el navegador
```

`compute-0-1` se cambia por el nodo que dé `squeue -u $USER -h -o "%N"`.

## Tests

- **Seguimiento de ficheros:** se escriben líneas a trozos (incluida una sin `\n`) y el lector las
  entrega enteras, una sola vez y en orden.
- **SSE:** con el panel arrancado en un puerto libre y un cliente httpx en streaming, las líneas que
  se añaden a `datos.jsonl` llegan como eventos `solicitud` en < 2 s. Al escribir `entorno.json` llega el evento `estado`.
- **`/api/tandas`:** lista las tandas en orden y con el estado correcto.
- **Sin interferencia:** una tanda con el servidor falso y el panel conectado produce el mismo
  `datos.json` (salvo tiempos) que sin panel. El panel nunca escribe en la carpeta de la tanda.
- **Proxy (si entra):** el stream que recibe el cliente es idéntico byte a byte al del servidor
  falso, y la medida del proxy coincide con la de `medidor medir` (±5 %) contra el mismo falso.

## Validación en el clúster (tú)

1. Lanzar el panel y la tanda de la fase 1, y abrir el panel desde el PC por el túnel: las
   solicitudes aparecen mientras se miden (criterio 1).
2. Repetir la tanda sin panel y comparar la TTFT p50 (criterio 2, ±2 %).
3. Cerrar el navegador y cortar el túnel a mitad de la tanda: la tanda termina igual (criterio 3).

## Instrucción para Claude Code (PC)

> Repo medidor-a100: lee `DESARROLLO.md`, `docs/fases/README.md` y `docs/fases/FASE-4.md`. Primero
> pregúntame la decisión del inicio (A panel web / B terminal, y si entra el proxy 4.4). Después propón
> los pasos numerados y espera aprobación; sin commit. El panel solo lee ficheros, usa solo la biblioteca
> estándar en el servidor y una página autocontenida sin CDN. Tests en verde en Windows sin red ni GPU.
