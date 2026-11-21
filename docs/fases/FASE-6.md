# Fase 6: gráficas de tesis y exportación

**Objetivo:** generar desde los JSON las gráficas de la tesis (PDF vectorial para LaTeX y PNG) y
empaquetar los resultados en un zip en `$HOME`, porque Jupyter no deja descargar carpetas.

**Criterio de aceptación:** las 7 gráficas salen en PNG y PDF a partir de una tanda real, y
`medidor exportar` deja un zip en `$HOME` que se puede descargar desde Jupyter.

**Requisito:** fases 1–5.

---

## Material existente que reutilizar

`referencias/generar_graficas.py` (13 figuras F01–F13 de lo medido hasta el 20-nov, en
`~/tfg-jose-ureta/graficas/`) y `referencias/exportar_resumen.sh`. **Reutilizar su estilo**
(tipografía, tamaños, colores) para que las figuras nuevas y las viejas encajen en la tesis.

## 6.1 `medidor graficas <dir> [<dir> …]`

Con matplotlib y un estilo único: serif, ancho de columna de LaTeX, PDF vectorial y PNG a 300 dpi.
Se escriben en `<dir>/graficas/` sin sobrescribir; si ya existen, en `graficas-<fecha>/`.

| # | gráfica | de dónde sale |
|---|---|---|
| 1 | TTFT vs. tokens de entrada (p50 con banda p10–p90), una línea por backend/modelo | `resumen.json` (añadir p10 a `estadisticos`) |
| 2 | prefill tok/s vs. tokens de entrada | `servidor_prefill_tok_s` (Ollama) / `prefill_tok_s_cliente` (vLLM), indicando cuál |
| 3 | TPOT y decode tok/s vs. concurrencia | `resumen.json` |
| 4 | throughput agregado vs. concurrencia (saturación) | fase 5 |
| 5 | CDF de ITL por punto | `datos.json` (`itl_s` de las solicitudes medidas) |
| 6 | serie temporal de la GPU (memoria, potencia, uso) con las fases de la tanda sombreadas | `telemetria.json` + `datos.json` |
| 7 | comparativa vLLM vs. Ollama, mismo modelo y mismo punto | dos tandas |

**Reglas:**
- Cada figura lleva al pie (en los metadatos del PDF y en un `graficas/indice.json`) la tanda, la
  fecha, el `n` de cada punto y el esquema.
- **Si un punto tiene errores**, se dibuja con su `n` real y se marca; no se interpola.
- **Las gráficas no dependen de un `$DISPLAY`**: backend `Agg`.

## 6.2 `medidor exportar`

- **Actualiza `resultados/indice.json`**: una entrada por tanda con experimento, fecha, backend,
  modelo, estado, nodo, job y los p50 principales. Es el "resumen global".
- **Crea `~/medidor-resultados.zip`** con todas las tandas: JSON, gráficas y logs.
  - Opción `--tandas`: solo unas tandas.
  - Usar `zipfile` de Python. Es lo que funcionó con `~/tfg-graficas.zip`.
  - El zip anterior se sustituye, **pero los datos no**: el zip es una copia.
- **No envolver nada en `timeout`**: el NFS lo cortó una vez y dejó matplotlib a medias.

## 6.3 Compatibilidad: `medidor exportar-vllm-bench <dir>`

Escribe además el formato de `vllm bench serve --save-result`, el que lee `resumen_pd.py`, para que
los scripts viejos sigan funcionando con los datos nuevos (DESARROLLO.md §5). El formato de referencia
está en `referencias/piloto-mig-0.6b/serve_*.json`.

## Tests

- **Las 7 gráficas** se generan desde una tanda sintética (fixture con `datos.json`, `resumen.json` y
  `telemetria.json` falsos) y existen en PDF y PNG. Basta con comprobar que el PDF empieza por `%PDF`
  y que el PNG tiene las dimensiones esperadas.
- **No se sobrescribe** una carpeta `graficas/` existente.
- **El zip** contiene lo esperado y el índice se rehace igual dos veces (es idempotente).
- **`exportar-vllm-bench`:** las claves y los tipos coinciden con un `serve_*.json` de referencia.

## Validación en el clúster (tú)

`medidor graficas` sobre las tandas reales de las fases 1–5 y `medidor exportar`. Después, bajar el
zip desde Jupyter y revisar las figuras en el PC. El criterio de calidad lo decides tú: si no son
"de tesis", se ajusta el estilo.

## Instrucción para Claude Code (PC)

> Repo medidor-a100: lee `DESARROLLO.md`, `docs/fases/README.md` y `docs/fases/FASE-6.md`, y el estilo de
> `referencias/generar_graficas.py`. Implementa `graficas`, `exportar` y `exportar-vllm-bench`. Propón los
> pasos numerados y espera aprobación; sin commit. Tests en verde en Windows sin red ni GPU.
