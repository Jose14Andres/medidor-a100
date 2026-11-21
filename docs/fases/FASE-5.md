# Fase 5: concurrencia

**Objetivo:** medir con varias solicitudes en vuelo a la vez (bucle cerrado con N fijos, o llegadas
Poisson a una tasa) y obtener el throughput agregado y la curva de saturación (E2 de la tesis).

**Criterio de aceptación:**
1. Con el servidor falso, el throughput medido es igual al teórico (±2 %).
2. En el clúster, las filas de **concurrencia 4** del piloto `piloto-mig-0.6b-20261027-155013` se reproducen (±5 %).

**Requisito:** fases 1–4. El panel de la fase 4 tiene que mostrar las solicitudes concurrentes;
dibujarlas solapadas es la gracia.

---

## Referencia (`referencias/piloto-mig-0.6b/RESUMEN.md`, vLLM, Qwen3-0.6B, MIG 1g.5gb)

| entrada | salida | conc. | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | salida tok/s | total tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 4 | 51,3 | 54,7 | 9,7 | 400,5 | 801,0 |
| 1024 | 128 | 4 | 183,9 | 221,4 | 12,4 | 288,9 | 2599,8 |

## Diseño

### 5.1 `src/medidor/carga.py`: planificadores

- **Cerrado (`modo = "cerrado"`):** N trabajadores `asyncio`; cada uno manda una solicitud y, al
  acabar, la siguiente. Se reparten `calentamiento + repeticiones` solicitudes por punto. **El
  calentamiento se hace con la misma concurrencia** y se marca, igual que en secuencial.
- **Poisson (`modo = "poisson"`, `tasa_rps`):** las llegadas siguen una exponencial con
  `random.Random(semilla)`, sin límite de solicitudes en vuelo. Se guarda la llegada **planificada** y
  la **real** de cada solicitud, para ver si el cliente se retrasa (el retraso del propio cliente no
  debe confundirse con la latencia del servidor).
- **Una sola conexión `httpx.AsyncClient`** compartida, con `limits` ≥ N conexiones. Si no, httpx
  serializa las solicitudes en silencio. Es un fallo típico: hay que comprobarlo con un test.
- La fase 1 limitaba a concurrencia 1 en `tanda.comprobar_fase`: esa restricción se quita.

### 5.2 Datos y resumen

- **Cada solicitud** añade `t_inicio_rel_s`, `t_fin_rel_s`, `en_vuelo_al_empezar` y, en Poisson, `t_llegada_planificada_s`.
- **Cada punto del resumen** añade:
  - `duracion_s`: de la primera solicitud medida que empieza a la última que acaba;
  - `throughput_salida_tok_s` = tokens de salida totales / duración;
  - `throughput_total_tok_s` = (entrada + salida) / duración;
  - `solicitudes_por_s`;
  - y, si la config define un SLO (`[slo] ttft_s = 1.0`, `tpot_s = 0.05`), `goodput` = % de
    solicitudes que lo cumplen.
- **El calentamiento no entra en la duración.** Durante el calentamiento hay solicitudes en vuelo
  que se solapan con las primeras medidas: se usa la ventana de las medidas y se documenta en las `notas`.

### 5.3 Servidor falso

Es un `ThreadingHTTPServer`, así que ya atiende en paralelo. Para el criterio 1 se añade
`Ajustes.max_en_vuelo` (un semáforo que simula la capacidad del servidor). Con latencias fijas, el
throughput teórico es `min(N, max_en_vuelo) × tokens_salida / e2e`.

## Tests

- **Cerrado:** con N = 4 y una latencia fija de 0,2 s, nunca hay más de 4 en vuelo y casi siempre
  hay 4; el throughput coincide con el teórico (±2 %).
- **Poisson:** con una semilla fija, las llegadas son reproducibles y la tasa media se acerca a
  `tasa_rps` en una tanda larga (estadístico, no exacto).
- **Conexiones:** con N = 8, el servidor falso recibe de verdad 8 conexiones a la vez (test del límite de httpx).
- **Errores bajo concurrencia:** la parada tras 3 errores seguidos cancela las solicitudes en vuelo
  sin perder las que ya terminaron.

## Validación en el clúster (tú)

El mismo arranque de vLLM que en la fase 2 (MIG, Qwen3-0.6B, puerto 8100), con `concurrencias = [1, 4]`.
Las filas c = 1 tienen que seguir dando lo de la fase 2 y las c = 4, lo de la tabla de arriba (±5 %).

## Instrucción para Claude Code (PC)

> Repo medidor-a100: lee `DESARROLLO.md`, `docs/fases/README.md` y `docs/fases/FASE-5.md`. Implementa la
> concurrencia (modos cerrado y Poisson) y el throughput agregado. Propón los pasos numerados y espera
> aprobación; sin commit. Tests en verde en Windows sin red ni GPU, con el servidor falso.
