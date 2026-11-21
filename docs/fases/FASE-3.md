# Fase 3: telemetría de GPU

**Objetivo:** muestrear la GPU cada 500 ms durante toda la tanda (memoria, % de uso, potencia,
temperatura, reloj SM) y poder alinear cada muestra con las solicitudes. Es la fuente de datos de
las gráficas de GPU de la fase 4 (en vivo) y de la fase 6 (tesis).

**Criterio de aceptación:** `telemetria.json` cubre toda la tanda, desde antes de la primera
solicitud hasta después de la última, y la memoria es > 0 durante las solicitudes.

**Requisito:** fases 1 y 2.

---

## Diseño

### 3.1 `src/medidor/telemetria.py`

El medidor corre en el propio nodo (modo A), así que la telemetría es un **proceso hijo** del medidor:

```
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu,
           power.draw,temperature.gpu,clocks.sm --format=csv,noheader,nounits -lms 500
```

- **Clase `Telemetria`**, que se usa como context manager asíncrono. Arranca el proceso, lee sus
  líneas en una tarea `asyncio` y, por cada muestra, guarda:
  - `t_utc`: reloj de pared del cliente al leerla;
  - `t_rel_s`: `perf_counter` respecto al inicio de la tanda, en la misma escala que las solicitudes;
  - `t_nvidia`: el timestamp de nvidia-smi;
  - los campos numéricos, con `[N/A]` → `null`.
- **Escritura incremental** en `telemetria.jsonl` (como `datos.jsonl`) y `telemetria.json` al final.
- **Parada:** al salir del context manager se termina **el propio proceso hijo** (`proc.terminate()`
  y espera con tiempo explícito). Esto no rompe la regla 1: es un proceso que lanza el medidor, no un
  servidor ni un job. Nunca se usa `pkill`.
- **El comando es un parámetro**: `Telemetria(argv=[...])`. Los tests pasan un nvidia-smi falso, que
  es un script de Python que imprime líneas CSV con un intervalo. Así funciona también en Windows.
- Si `nvidia-smi` muere a mitad de la tanda, queda registrado en `entorno.json`
  (`telemetria.estado = "cortada a los X s"`) y la tanda **sigue**: la telemetría no invalida las
  medidas. El resumen lo indica.

### 3.2 Alineación con las solicitudes

Cada solicitud ya tiene `t_inicio_utc` y, con esta fase, también `t_inicio_rel_s` (perf_counter
respecto al inicio de la tanda). Para cada punto, el resumen añade un bloque `gpu`: memoria máxima,
potencia media y máxima, % de uso medio y temperatura máxima **durante las solicitudes medidas** del
punto, usando solo las muestras dentro de [inicio, fin] de cada solicitud.

Y en `entorno.json`, la **línea base** de la GPU en reposo: unos 5 s de muestras antes de la carga del modelo.

### 3.3 MIG (DESARROLLO.md §12, decisión abierta 4)

En un trozo MIG, `utilization.gpu` y `power.draw` salen `[N/A]` o son de la GPU padre. Propuesta para decidir:

- **Por defecto:** guardar lo que dé nvidia-smi y marcar `"mig": true` en la telemetría. Las
  gráficas de uso y potencia no se dibujan en MIG.
- **Si `dcgmi` existe en el nodo** (`which dcgmi`): ofrecer `telemetria.fuente = "dcgm"` con
  `dcgmi dmon -e 1002,1003,1004,155 -d 500` (SM activo, ocupación, tensor, potencia) por instancia MIG.
  Antes hay que comprobar en el clúster que `dcgmi` está y que tienes permiso.

### 3.4 Config

`[telemetria]` ya existe (`activar`, `intervalo_ms`). Se añade `fuente = "nvidia-smi"` (o `"dcgm"`).
Con `activar = true` y sin nvidia-smi, la tanda **no** empieza: falla una guardia nueva **G10**
("telemetría pedida pero no disponible").

## Tests

- `tests/nvidia_smi_falso.py`: imprime N muestras cada X ms, con opciones para `[N/A]` (MIG) y para morir a la mitad.
- Las muestras cubren la tanda, el intervalo medio queda a ±20 % de `intervalo_ms` y la escritura es incremental.
- La alineación: con muestras sintéticas y solicitudes sintéticas, el bloque `gpu` del punto da los valores esperados.
- Si la telemetría se corta, la tanda sigue y el estado queda registrado.
- Al acabar no queda ningún proceso hijo vivo (`proc.returncode is not None`).

## Validación en el clúster (tú)

Repetir la tanda de la fase 1 (`v1-validacion-36268.toml` con `[telemetria] activar = true`) y comprobar:

1. La primera muestra es anterior a la primera solicitud y la última, posterior a la última.
2. La memoria es > 0 en todas las muestras durante las solicitudes (≈ 24,8 GB con nemotron-3-nano).
3. **Los números de la fase 1 no cambian** (±5 %): la telemetría no debe perturbar la medida.

## Instrucción para Claude Code (PC)

> Repo medidor-a100: lee `DESARROLLO.md`, `docs/fases/README.md` y `docs/fases/FASE-3.md`. Implementa la
> telemetría de GPU como proceso hijo con el comando inyectable (los tests usan un nvidia-smi falso en
> Python). Propón los pasos numerados y espera aprobación; sin commit. Tests en verde en Windows sin GPU.
