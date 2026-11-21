# Fase 7: chat y comparativas

**Objetivo:** `medidor chat` para probar un modelo a mano viendo TTFT y TPOT de cada respuesta (y
guardándolo en JSON), y comparativas entre tandas: vLLM frente a Ollama, un modelo frente a otro, un
día frente a otro.

**Criterio de aceptación:** `medidor chat` sustituye a `experimentos/chat_stream.py`, y una
comparativa entre dos tandas reales sale en tabla y en gráfica.

**Requisito:** fases 1–6.

---

## 7.1 `medidor chat -c <config>`

Toma como referencia `referencias/chat_stream.py` (stdlib, TTFT y TPOT por respuesta).

- **Usa la plantilla de chat:** `/api/chat` en Ollama y `/v1/chat/completions` en vLLM. Es la
  diferencia con la tanda, que mide en `raw` (DESARROLLO.md §12.2). El historial de la conversación se
  manda en cada turno.
- **Antes de empezar pasa las mismas guardias** (G1, G2, G3, G5). No tiene sentido chatear con un
  Ollama caído a CPU sin saberlo.
- **Tras cada respuesta** imprime: tokens de entrada (de toda la conversación), TTFT, TPOT, decode
  tok/s y `finish_reason`.
- **Guarda la sesión** en `resultados/chat-<fecha>/datos.jsonl` con el esquema v1: mismo formato que
  las tandas, `punto` = tokens reales y `"modo": "chat"`. Así el panel de la fase 4 la muestra en vivo.
- **Órdenes:** `/salir`, `/nuevo` (borra el historial), `/stats` (resumen de la sesión), `/sistema <texto>`.
- **El texto de las respuestas** se guarda solo con `--guardar-texto`. Por defecto no: los datos de
  la tesis son métricas y no hace falta guardar contenido.

## 7.2 `medidor comparar <dir_a> <dir_b> [...]`

Amplía el comparador de la fase 1 (1.1) a tandas del medidor entre sí:

- Empareja los puntos por (entrada, salida, concurrencia) y, si no coinciden exactamente, por tokens reales (±2 %).
- Saca una tabla con p50 y p90 de TTFT, TPOT, prefill y decode tok/s y throughput, con la diferencia
  en %. Indica qué tanda tenía más loadavg y en qué nodo y GPU corrió cada una (de `entorno.json`).
- Escribe `comparacion.json` y la **gráfica 7** de la fase 6.
- **Avisa** cuando se comparan cosas no comparables: distinto `num_ctx`, distinta GPU (MIG frente a
  A100 entera), modo `raw` frente a chat. Avisa, pero no bloquea.

## Tests

- **Chat** contra el servidor falso, añadiéndole `/api/chat` y `/v1/chat/completions`: el historial
  crece en cada turno, las métricas por turno son coherentes, la sesión se guarda en JSONL y
  `--guardar-texto` se respeta.
- **Comparar:** emparejamiento exacto y aproximado, los avisos de no comparables y la escritura sin sobrescribir.

## Validación en el clúster (tú)

- **Chat:** una sesión con nemotron-3-nano en Ollama. Las TTFT y TPOT que imprime tienen que ser del
  orden de lo de la tanda (decode ~150 tok/s).
- **Comparar:** las tandas de la fase 1 (Ollama) y de la fase 2 (vLLM). Si hay un modelo en los dos
  backends, sale la comparativa directa.

## Instrucción para Claude Code (PC)

> Repo medidor-a100: lee `DESARROLLO.md`, `docs/fases/README.md` y `docs/fases/FASE-7.md`, y
> `referencias/chat_stream.py`. Implementa `medidor chat` y amplía `medidor comparar`. Propón los pasos
> numerados y espera aprobación; sin commit. Tests en verde en Windows sin red ni GPU.
