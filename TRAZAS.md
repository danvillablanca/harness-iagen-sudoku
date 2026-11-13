# Generación de trazas con Qwen3.5-9B (para SFT / DPO del 2B)

Plan para generar trazas con `Qwen/Qwen3.5-9B` sobre `data/sudoku/train.jsonl`, que luego
sirven para entrenar `Qwen/Qwen3.5-2B` con SFT y/o DPO. Este documento cubre solo la
**generación**; la construcción de los datasets de SFT/DPO y el entrenamiento van después.

## Decisiones

| Tema | Decisión |
|---|---|
| Modelo profesor | `Qwen/Qwen3.5-9B` |
| Modo de razonamiento | `--no-thinking` (sin modo think) |
| `--max-output-tokens` | 4096 |
| Tareas | `train.jsonl`: 4x4 (easy, medium, hard) y 6x6 (easy, medium). Sin 6x6 hard ni 9x9 |
| Muestreo | Predeterminado de Qwen sin thinking: temperatura 0.7, top_p 0.8 |
| Muestras por tarea | `n = 8` (a ajustar con el piloto) |
| Few-shot | Piloto 0 vs 2 shots; por defecto 2 shots, formato `steps` (ver abajo) |

`train.jsonl` tiene 50 tareas por (tamaño, nivel), así que el subconjunto son **250 tareas**
(150 de 4x4 y 100 de 6x6). Con `n = 8` son 2000 muestras. Todas las tareas tienen solución
única demostrada y se resuelven por casillas con un único candidato, por lo que el
verificador (`reward`) es exacto.

## ¿Few-shot o no?

Recomendación: **generar con 2 shots en formato `steps`, pero guardar el prompt sin shots**
(destilación del prompt).

- **A favor de los shots.** Sin ejemplos, el 9B sin thinking puede responder con la rejilla
  directa o con un razonamiento libre de formato variable. Con shots responde
  `Step k: rXcY = v` más la rejilla y termina en `<answer>`. Esa traza es corta, uniforme y
  aprendible por un modelo de 2B, y reproduce el formato que ya usa `make_shots`.
- **El prompt de entrenamiento no lleva shots.** El 2B se evalúa con el prompt base. Cada
  ejemplo de SFT/DPO será `prompt de train.jsonl → completion`, de modo que el 2B aprende a
  producir el estilo de los shots sin necesitar los ejemplos en contexto. Las trazas siguen
  cumpliendo el contrato de `parse_answer`, porque los shots no cambian el formato de
  respuesta.
- **En contra.** Los shots alargan el prompt (límite 16384 tokens, ya automático con
  `--shots`) y hacen más lenta la generación. También puede copiarse el estilo de los shots
  aunque sea subóptimo.
- **Piloto antes de decidir.** Correr 0 y 2 shots sobre 5 tareas por nivel con `n = 8` y
  comparar con `baseline.py summary`: tasa de aprobación (pass@1), tasa de truncamiento,
  media de tokens. Se usa 2 shots salvo que 0 shots apruebe igual o más y con trazas de
  formato igualmente limpio. El formato `compact` es una alternativa más corta si las trazas
  de 6x6 `medium` truncan a 4096; `seen` es más largo.

## Cambio de código (hecho)

`baseline.py run` ya muestrea `n` completions por tarea, las puntúa con `reward_parts`
(guarda `reward`, `reward_parts`, `finish_reason`, `generated_tokens` y `completion`) y
reanuda si se interrumpe. Solo falta excluir **6x6 hard**: `--sizes` filtra por tamaño, pero
no por nivel.

- Implementado: `--skip SIZE:LEVEL [...]` (p. ej. `--skip 6:hard`), aplicado en
  `select_tasks`, con pruebas en `test_baseline.py`.
- Para este plan: `--sizes 4 6 --skip 6:hard`.

## Procedimiento

1. **Entorno** (una vez, en el nodo de login; ver `CLAUDE.local.md`):
   `module load python/3.13 cuda/12.9`, crear `.venv`, instalar `requirements.txt`,
   `export HF_HOME=$SCRATCH/hf`. El 9B en bf16 ocupa unos 18 GB y cabe en una L40S (48 GB).
   **Nunca correr inferencia en el nodo de login**: usar `sbatch` con
   `--account=aip-sheilam --gres=gpu:l40s:1`.
2. **Filtro de niveles** (`--skip`, ya implementado en `baseline.py` con pruebas). Tests:
   `python -m unittest discover -s Laboratorio/Qwen3.5 -p "test_*.py"`.
3. **Piloto** (≤ 3 h, una sola L40S), un directorio de salida por variante:
   ```sh
   COMMON="--tasks data/sudoku/train.jsonl --model-id Qwen/Qwen3.5-9B --no-thinking \
     --max-output-tokens 4096 --sizes 4 6 --skip 6:hard --per-level 5 --n 8"
   python Laboratorio/Qwen3.5/baseline.py run $COMMON --out results/pilot/0shot
   python Laboratorio/Qwen3.5/baseline.py run $COMMON --shots 2 --out results/pilot/2shot
   python Laboratorio/Qwen3.5/baseline.py summary results/pilot/*/*.jsonl
   ```
   Del piloto sale también el tiempo por muestra, para dimensionar la corrida completa.
   Se lanza como dos jobs en paralelo, uno por variante:
   `sbatch --job-name=pilot-0shot scripts/traces_pilot.sbatch` y
   `sbatch --job-name=pilot-2shot --export=ALL,SHOTS=2 scripts/traces_pilot.sbatch`.
4. **Corrida completa** con la variante elegida y todas las tareas (sin `--per-level`):
   ```sh
   python Laboratorio/Qwen3.5/baseline.py run \
     --tasks data/sudoku/train.jsonl --model-id Qwen/Qwen3.5-9B --no-thinking \
     --max-output-tokens 4096 --sizes 4 6 --skip 6:hard --n 8 \
     --shots 2 --out results/traces/r0
   ```
   Lanzar con `SHOTS=2 sbatch scripts/traces.sbatch` (variables `SHOTS`, `ROUND`, `SEED`,
   `SHOT_SEED`). Si no cabe en la ventana del job, usar `--time` hasta 12 h (partición b2) o repetir el mismo comando:
   reanuda donde quedó.
5. **Revisar la cobertura** con `summary`: por (tamaño, nivel) la tasa de aprobación, la tasa
   de truncamiento y la media de tokens. Esto indica cuántas tareas tienen aprobadas (SFT) y
   cuántas tienen aprobadas y fallidas a la vez (pares DPO).
6. **Rondas extra si falta cobertura** (típicamente 6x6 medium): volver a correr con otro
   `--seed` y, si se quiere diversidad de shots, otro `--shot-seed`, siempre en un `--out`
   nuevo (`results/traces/r1`, ...). Reanudar sobre el mismo archivo con otros ajustes se
   rechaza a propósito (`RUN_KEYS`).

## Qué queda registrado por muestra

`task_id`, `size`, `level`, `empty_cells`, `sample`, `reward`, `reward_parts` (format,
clues_frac, fill, solved, exact), `finish_reason`, `generated_tokens`, `completion` y todos
los ajustes de la corrida. El prompt no se duplica: se recupera de `train.jsonl` por
`task_id`. Los shots tampoco se guardan; se reconstruyen con `--shot-seed`, pero no entran
en los datos de entrenamiento.

## Después de generar (fuera de este documento)

- **SFT:** muestras con `reward == 1.0` y `finish_reason` distinto de truncamiento,
  1–3 completions distintas por tarea (preferir las más cortas), prompt sin shots.
- **DPO:** por tarea, un par (aprobada, fallida). Preferir rechazadas con `format = 1` y
  `solved = 0` (rejillas casi correctas); añadir algunas truncadas o de formato roto.
- **Evaluación:** el 2B se evalúa con `baseline.py` sobre `eval.jsonl` con los mismos
  ajustes que su baseline. Medir aparte 6x6 hard y 9x9 para ver si el 2B generaliza o
  retrocede en lo que no vio.

## Riesgos

- **Desbalance.** Es probable que 4x4 apruebe casi todo y 6x6 medium poco; las trazas
  quedarán sesgadas a lo fácil. Mitigar limitando las completions por tarea y, si hace
  falta, generando un split `train` más grande con `dataset.py`.
- **Truncamiento a 4096 tokens** en 6x6 medium (hasta 18 pasos con rejilla completa por
  paso). Se cuenta como fallo; mirar `finish_reason` en el piloto antes de la corrida larga.
- **Mismo generador que el test.** Las semillas impiden compartir puzzles entre splits, pero
  conviene verificar que ningún prompt de `train.jsonl` aparezca en `eval.jsonl`/`test.jsonl`.
