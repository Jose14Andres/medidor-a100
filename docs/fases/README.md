# medidor-a100: plan por fases

Objetivo final: **conectar el medidor en el clúster de la A100 y ver en tiempo real, desde el PC,
las mediciones que se hagan con Ollama o con vLLM**, con todo guardado en JSON para la tesis.

El diseño general está en [`../../DESARROLLO.md`](../../DESARROLLO.md). Aquí hay un documento por
fase, pensado para desarrollar desde el PC (con Claude Code u otro editor) sin abrir terminales en el
clúster salvo para validar con la GPU.

## Cuántas fases faltan

Son 8 fases en total (0 a 7). El orden se cambió el 20-nov-2026 para llegar antes a la vista en
tiempo real: ahora va en la fase 4, antes que la concurrencia.

| fase | qué entrega | estado (20-nov-2026) | ¿necesaria para el objetivo? |
|---|---|---|---|
| 0 | esqueleto, `medidor entorno`, servidor falso, CI | **hecha** | sí |
| [1](FASE-1.md) | Ollama secuencial + guardias + JSON | **código hecho (81 tests)**; falta validar en la A100 | sí |
| [2](FASE-2.md) | vLLM secuencial | pendiente | sí ("o los LLM") |
| [3](FASE-3.md) | telemetría GPU (nvidia-smi cada 500 ms) | pendiente | sí (GPU en vivo) |
| [4](FASE-4.md) | **vista en tiempo real** (panel web por túnel SSH) | pendiente | **es el objetivo** |
| [5](FASE-5.md) | concurrencia (cerrado y Poisson), throughput | pendiente | para la tesis |
| [6](FASE-6.md) | gráficas de tesis + exportación (zip en `$HOME`) | pendiente | para la tesis |
| [7](FASE-7.md) | `medidor chat` + comparativas entre tandas | pendiente | para la tesis |

**Para lo que pides (ver en tiempo real con Ollama y vLLM) basta con llegar a la fase 4**: cerrar
la validación de la 1 y hacer la 2, la 3 y la 4. Las fases 5, 6 y 7 completan la herramienta para la
tesis.

## Cómo se trabaja: PC ↔ clúster

```
PC (Windows)                                    clúster CEDIA (nodo GPU)
────────────                                    ────────────────────────
editar código + tests con servidor falso         git pull
pytest en verde (sin red ni GPU)                 medidor comprobar / medir   ← validación con la A100
git commit + git push  ───── GitHub ─────►       (el panel de la fase 4 se ve desde el PC por túnel)
```

- **En el PC:** todo el desarrollo. Los tests no necesitan red ni GPU: usan
  `tests/servidor_falso.py` (imita vLLM y Ollama) y extractos reales de logs en `tests/datos/`.
- **En el clúster:** solo `git pull` y la validación de cada fase con la GPU (los comandos exactos
  están en cada documento). El clúster no tiene credenciales de GitHub: el push se hace siempre desde el PC.

### Primer paso en el PC: traer la fase 1

La fase 1 está en un commit local del clúster que no se pudo subir. Viene en
`medidor-a100-fase1.bundle`, dentro del zip:

```powershell
cd ruta\a\tu\clon\medidor-a100          # el clon donde hiciste la fase 0
git pull ruta\a\medidor-a100-fase1.bundle main
git log --oneline -3                    # deben salir los commits de la fase 1 y de estos documentos
git push origin main
python -m venv .venv
.venv\Scripts\pip install --require-hashes -r requirements.lock
.venv\Scripts\pip install --no-deps -e .
.venv\Scripts\python -m pytest -q       # 81 tests en verde (en Windows aún no se han ejecutado)
```

Si no tienes el clon: `git clone ruta\a\medidor-a100-fase1.bundle medidor-a100`, luego
`git remote set-url origin https://github.com/Jose14Andres/medidor-a100.git` y `git push origin main`.

Después, en el clúster (una sola vez): `cd ~/medidor-a100 && git pull` (o `git fetch && git reset --hard origin/main`
si el pull se queja; el contenido es el mismo).

### Material de referencia (`referencias/` en el zip, no está en el repo)

Código y datos del clúster que las fases tienen que reutilizar o reproducir. No van al repo porque
son datos de la tesis:

| fichero | para qué fase |
|---|---|
| `tfgbench/*.py` (cliente streaming vLLM, tokens, esquema) | 2 |
| `medir_ollama.py`, `resumen_36268.json` | 1 (validación) |
| `piloto-mig-0.6b/RESUMEN.md`, `serve_*.json`, `piloto_mig_06b.toml` | 2 y 5 (validación) |
| `servir_vllm.sh` | 2 |
| `chat_stream.py` | 7 |
| `generar_graficas.py`, `exportar_resumen.sh` | 6 |

## Reglas que valen para todas las fases

Son las reglas del TFG llevadas al código (DESARROLLO.md §6). Si una fase las rompe, no está terminada.

1. **No mata nada.** Nunca `kill`, `scancel`, `pkill`. Ojo también con `kill $(pgrep -f ...)`: mata la
   propia shell. Si hay que parar un servidor, lo hace la persona después de leer las últimas 40 líneas de su log.
2. **Los tiempos de espera no se suben para "arreglar".** Son parámetros explícitos y quedan en `entorno.json`.
   Si salta uno, se informa de dónde estaba la solicitud.
3. **"Servidor listo" exige salud HTTP y memoria GPU > 0** (guardias G1 y G2).
4. **Solo números medidos.** El resumen no extrapola ni rellena; los errores se cuentan, no se ocultan.
5. **No sobrescribe.** Cada tanda escribe en una carpeta nueva; nunca se tocan `tesis/resultados/`,
   `resultados/` ni `ollama/resultados/`.
6. **Si una validación no coincide, se para y se investiga** antes de seguir con la fase siguiente.
7. **Todo en JSON con `"esquema": "medidor-a100/v1"`.** Un cambio incompatible del esquema sube a v2.
8. **Tests sin red ni GPU**, deterministas, en Windows y en Linux. `./ci.sh` (Linux) o `pytest -q` (Windows).
9. **Textos, mensajes y documentación en español.**

## Cómo usar estos documentos con Claude Code en el PC

Cada `FASE-N.md` termina con una **instrucción para Claude Code** que puedes pegar tal cual. La idea
es ir por pasos numerados: Claude propone, tú apruebas, Claude implementa y deja los tests en verde,
y tú haces commit y push. La validación con la GPU la ejecutas tú en el clúster con los comandos del
documento y le pasas el resultado (o el `resumen.json`).
