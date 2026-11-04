# harness-iagen-sudoku
Harness de Sudoku para el curso IIC3694-1 Inteligencia Artificial Generativa.

# Compilar el LATEX:
cd informe_IIC3694
latexmk -pdf -interaction=nonstopmode -halt-on-error informe.tex

# Requerimientos
Python 3.13.x de 64 bits, se recomienda entorno virtual.
Ejecutar desde la raíz:
python -m pip install -r requirements.txt

Hardware recomendado:

| Comprobación | CPU | Mac / MPS | NVIDIA / CUDA |
| --- | --- | --- | --- |
| Python | 64 bits | 64 bits, ARM64 nativo | 64 bits |
| CPU lógicas detectadas | 4 o más | 2 o más | 2 o más |
| RAM total utilizable | 15 GiB o más | 15 GiB o más, unificada | 15 GiB o más |
| RAM disponible, sin sumar swap | 12 GiB o más | 8 GiB o más | 6 GiB o más |
| VRAM | No aplica | Comparte la RAM | Al menos 7,5 GiB totales y 6 GiB libres |
| Límite de trabajo recomendado por Metal | No aplica | 6 GiB o más; es un límite orientativo, no memoria libre | No aplica |
| Tipo de los pesos cargados | FP32 | FP16 | FP16 |
## Uso del harness

Ejecutar desde la raíz con Python 3.13:

```sh
python Laboratorio/Qwen3.5/main.py
```

En `main.py`, `ModelConfig` permite configurar el modelo, backend (`auto`,
`mlx`, `cuda`, `mps` o `cpu`), modo thinking y límites de tokens. Selecciona
un caso con `prompts["sudoku9x9"]["hard"]`; esta etiqueta aún no ha sido
calibrada experimentalmente. Añade casos al registro en `prompts.py`.

### Argumentos de línea de comandos

| Argumento | Descripción | Por defecto |
| --- | --- | --- |
| `--thinking` / `--no-thinking` | Activa o desactiva el modo de razonamiento de Qwen | activado |
| `--max-output-tokens N` | Límite de tokens generados | 8192 |
| `--max-prompt-tokens N` | Límite de tokens de entrada | 2048 (16384 si `--shots > 0`) |
| `--empty-cells N` | Número de celdas vacías; elige al azar cuáles borrar de la solución | puzzle fijo de `sudoku.json` |
| `--seed N` | Semilla para elegir las celdas vacías | 0 |
| `--shots N` | Número de ejemplos resueltos (few-shot) antes del puzzle | 0 |
| `--shot-seed N` | Semilla para generar los puzzles de los ejemplos | 0 |

Con `--shots N`, cada ejemplo es un puzzle aleatorio distinto del de prueba
(8 celdas vacías), resuelto una celda por vez: cada paso (`Step k: rXcY = v`)
muestra la grilla completa, y la respuesta termina en `<answer>[...]</answer>`.
Los shots solo añaden ejemplos: el prompt del ejercicio es idéntico con o sin
shots. Se permiten pasos, explicaciones y grillas intermedias antes de la
respuesta final; no se exige un procedimiento específico.

El contrato de respuesta es el mismo en ambos casos:

```text
Explicaciones o pasos opcionales...
<answer>
[[...], ...]
</answer>
```

Debe existir exactamente un bloque final, con una matriz JSON de enteros sin
comentarios ni Markdown dentro. Solo se permite espacio en blanco después del
cierre. Las etiquetas no deben aparecer en el razonamiento ni en los ejemplos
que el modelo incluya en su propia respuesta. La corrección se basa en el
tablero final, no en la validez de cada paso de la explicación.

Al terminar, el harness imprime tiempos, tokens generados, motivo de término,
número de shots, el puzzle, la solución esperada, la solución del modelo, la
entrada completa al modelo (shots incluidos) y su salida completa.
También muestra cada validación, los problemas del parser, la corrección
matemática del tablero y la recompensa. `PASSED` requiere tanto una solución
válida como el formato requerido. Una matriz sin etiquetas se puede recuperar
para diagnóstico, pero no recibe recompensa.

```sh
python Laboratorio/Qwen3.5/main.py --no-thinking --empty-cells 20 --seed 1 --shots 2
```

`model.generate(case.prompt)` devuelve texto; `model.generate_with_stats(...)`
devuelve también los tokens y el motivo de término. Ambas aceptan las opciones
por llamada `max_tokens`, `enable_thinking` y `shots` (pares usuario/asistente
antes del prompt), que reemplazan los valores de la configuración. `timed(funcion, argumentos...)` devuelve `(resultado, segundos)`
y `evaluate(resultado.text, case)` verifica la respuesta; para el texto de
`generate`, pasa directamente ese texto a `evaluate`.

## Recompensa y verificación

`Laboratorio/Qwen3.5/reward.py` expone `reward(prompt, completion) -> float`.
Se importa igual que el resto de los módulos del harness, agregando
`Laboratorio/Qwen3.5` al path de Python si se usa desde otro directorio.
No carga modelos ni depende de PyTorch.

```python
from prompts import make_case
from reward import reward

case = make_case("sudoku9x9", "hard", empty_cells=1, seed=1)
score = reward(case.prompt, respuesta_del_modelo)
```

La versión actual es binaria: `1.0` para una solución válida en el formato
requerido y `0.0` en otro caso. Se comprueban filas, columnas, bloques, valores
y conservación de pistas, sin comparar con `expected_solution`. Por eso otra
solución válida recibe el mismo puntaje. El argumento `prompt` debe ser el
texto emitido por `build_prompt`, con dimensiones, bloques y una sección
`Puzzle:` con la matriz JSON. Un prompt mal configurado lanza `ValueError`;
una respuesta inválida recibe cero. Los pasos no se puntúan.

El crédito parcial y la ponderación por dificultad siguen pendientes. Una
posible medida parcial es la fracción de casillas originalmente vacías que
coinciden con la solución válida más cercana a la respuesta, considerando
todas las casillas contra una misma solución. Esta definición admite múltiples
soluciones, pero requiere una búsqueda de optimización; no se aproxima con la
cantidad de casillas llenas ni con coincidencias contra una referencia arbitraria.

## Dataset

`Laboratorio/Qwen3.5/dataset.py` genera particiones `train`, `eval` y `test` en
JSONL, con tableros de 4x4, 6x6 y 9x9 y niveles `easy`, `medium` y `hard` según
la cantidad de casillas vacías (`LEVELS`; aún no calibrados con el modelo).
Adapta el generador de Sudoku de Reasoning Gym: crea una solución aleatoria y
borra casillas una por una, deshaciendo cada borrado que permita una segunda
solución.

```sh
python Laboratorio/Qwen3.5/dataset.py --out data/sudoku
python Laboratorio/Qwen3.5/dataset.py --out data/sudoku9 --sizes 9 --train 200 --eval 50 --test 50 --seed 1
```

| Argumento | Descripción | Por defecto |
| --- | --- | --- |
| `--out DIR` | Carpeta de salida | `data/sudoku` |
| `--sizes N...` | Tamaños de tablero (4, 6, 9) | `4 6 9` |
| `--train/--eval/--test N` | Tareas por tamaño y nivel | 50 / 10 / 10 |
| `--seed N` | Semilla del dataset | 0 |
| `--unique` / `--no-unique` | Exigir solución única | exigida |

Cada línea contiene `id`, `split`, `size`, `box_rows`, `box_columns`, `level`,
`empty_cells` (contadas en el puzzle generado), `unique_solution`,
`deducible_by_singles` (si se resuelve llenando solo casillas con un único
candidato), `prompt` (de `build_prompt`), `puzzle` y `solution`. Cada tarea usa
su propia semilla, por lo que los mismos argumentos producen los mismos
archivos; ningún puzzle se repite entre particiones, y aumentar `--train` no
cambia `eval` ni `test`. `reward(record["prompt"], respuesta)` funciona
directamente, y `case_from_record(record)` entrega un `PromptCase` para
`evaluate` y `print_result`.

Tests (con las dependencias del proyecto):

```sh
python -m unittest discover -s Laboratorio/Qwen3.5 -p "test_*.py"
```

## Pasos 3 y 4 del Desafío: datos derivados y línea base

El paso 3 prepara datos para entrenar después: **el 9B es el profesor y el 2B
es el modelo objetivo**. No se ajustan pesos en esta entrega. `origin/baseline`
de Lilly aporta el estimador pass@k y el muestreo múltiple; esta integración añade
procedencia, reanudación por intento, auditoría y ejecución de ambos pasos.

### Ejecutar el flujo en el Mac

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt -r Laboratorio/Qwen3.5/requirements-mlx.txt
# Solo si todavía no existe esta carpeta; el generador evita sobrescribir datasets.
python Laboratorio/Qwen3.5/dataset.py --out data/sudoku-final --response-style final
# 9 tareas de train y 9 de eval: una por tamaño/nivel, 4 respuestas por tarea.
HF_HUB_OFFLINE=1 python Laboratorio/Qwen3.5/challenge.py --out results/mi-piloto
# Reanudar con TODOS los mismos argumentos, agregando --resume:
HF_HUB_OFFLINE=1 python Laboratorio/Qwen3.5/challenge.py --out results/mi-piloto --resume
```

`challenge.py` ejecuta secuencialmente: trazas del 9B sobre train → baseline del
9B sobre eval → baseline del 2B sobre las mismas tareas de eval → auditoría sin
modelos. Cada modelo vive en su propio proceso y se libera antes del siguiente.
Usa los pesos originales `Qwen/Qwen3.5-2B` y los ya instalados de
`mlx-community/Qwen3.5-9B-4bit`; no contempla el 27B. `HF_HUB_OFFLINE=1` impide
nuevas descargas. El tamaño y la precisión cambian a la vez: esta comparación
mide esas dos configuraciones, no aísla el efecto del número de parámetros.

El piloto utiliza temperatura 0.7, top-p 0.8, top-k 20, penalización de presencia
1.5, semilla 42, `thinking=False` y un máximo de 1024 tokens. Son parámetros
explícitos de este experimento, con muestreo positivo para obtener diversidad.
La [ficha oficial de Qwen](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices)
describe los parámetros de inferencia y su efecto sobre la repetición.

La variante `--response-style final` conserva exactamente los puzzles generados
con la misma semilla y pide solo el tablero final. El pipeline además usa
`--answer-prefix`: añade `<answer>\n[` al inicio del turno del asistente.
Este prefijo **solo fija formato**; todos los valores de las celdas y el cierre
son generados por el modelo. Se contabiliza como entrada, no como tokens generados.
`completion` contiene la respuesta completa (prefijo más continuación) que evalúa
el verificador. La configuración y el código quedan registrados en el manifiesto.
Por ello son métricas del modelo con este harness y este prefijo. No deben mezclarse
con las corridas previas sin prefijo, con thinking o con otros presupuestos de tokens.
Las trazas de este modo son respuestas verificadas, no cadenas de pensamiento validadas.

Para ampliar la muestra sin usar el conjunto test para ajustar decisiones:

```sh
# 5 tareas por tamaño/nivel y 8 respuestas: 45 tareas por modelo.
python Laboratorio/Qwen3.5/challenge.py --out results/calibracion-eval \
    --per-level 5 --samples 8
# Todas las tareas train/eval, 8 respuestas por tarea: corrida larga, reanudable.
python Laboratorio/Qwen3.5/challenge.py --out results/completo --profile full
```

`--dry-run` verifica y muestra el plan sin cargar pesos ni crear resultados.
`--split test` permite la evaluación final una vez fijado el protocolo; conservar
`eval` para calibrar. El piloto no sustituye una evaluación amplia ni demuestra
que las etiquetas easy/medium/hard estén calibradas.

### Archivos derivados del profesor

`traces.py` acepta exclusivamente tareas de train, sin IDs o prompts duplicados.
El modelo recibe el prompt público; nunca la solución de referencia. Guarda cada
intento antes de exportar y reconstruye los archivos derivados al reanudar.

```sh
python Laboratorio/Qwen3.5/traces.py --input data/sudoku-final/train.jsonl \
    --out results/trazas-9b --model mlx-community/Qwen3.5-9B-4bit --backend mlx \
    --per-level 1 --k 4 --no-thinking --answer-prefix --max-output-tokens 1024
```

- `manifest.json`: hash de entrada, tareas, modelo y revisión de caché, parámetros,
  versiones, hashes del código y procedimiento de filtrado.
- `candidates.jsonl`: todos los intentos, recompensa, diagnóstico de formato,
  verificaciones, semillas, tokens y tiempos. No se eliminan fallos.
- `sft.jsonl`: prompt y completion verificada, identificadores de procedencia y
  `messages` de chat. Elimina textos idénticos dentro de cada tarea.
- `dpo.jsonl`: prompt, chosen, rejected y IDs; la primera correcta se combina con
  cada incorrecta distinta de la **misma** tarea. No inventa rechazos ni soluciones.
- `summary.json`: cobertura por tamaño/nivel, trazas, pares, truncamientos y tareas
  completas sin éxito o sin rechazo. Cero pares DPO significa que falta diversidad.

Repetir el mismo comando con `--resume` continúa desde el intento pendiente.
`--resume --export-only` reconstruye los derivados sin cargar pesos y vuelve a
verificar las recompensas guardadas. Una escritura final truncada se respalda en
`.partial` antes de recuperarse; un JSON completo corrupto provoca error.
Cambiar datos, parámetros, código relevante o entorno exige una carpeta nueva.
Las semillas se derivan de SHA256(seed, task_id, intento); reanudar no las cambia.
No se garantiza identidad numérica entre hardware o versiones diferentes.

Con thinking habilitado (sin prefijo), se restaura `<think>` en los exports si
la plantilla lo puso en el prompt y la continuación solo contiene `</think>`.
Se conserva la respuesta original en candidatos. La recompensa comprueba únicamente
el tablero final y el contrato de formato, no cada paso de razonamiento.

### Métricas de varios modelos

`baseline.py` solo acepta eval o test, genera intentos independientes con
checkpoint individual y registra el mismo protocolo por modelo. La modalidad
por lotes de Lilly se conserva en `Qwen.generate_samples`; el baseline usa una
muestra por llamada para que las semillas no dependan del lote o la reanudación.

```sh
python Laboratorio/Qwen3.5/baseline.py run --tasks data/sudoku-final/eval.jsonl \
    --model-id Qwen/Qwen3.5-2B --backend mlx --no-thinking --answer-prefix \
    --per-level 1 --n 4 --seed 42 --max-output-tokens 1024 --out results/baseline
python Laboratorio/Qwen3.5/baseline.py summary results/baseline/*.jsonl \
    --k 1 2 4 --out results/baseline/metrics.json
```

Para n intentos y c aciertos, se estima `pass@k = 1 - C(n-c,k)/C(n,k)`,
con k ≤ n, y se promedia **por tarea**, según
[Chen et al. (2021)](https://arxiv.org/abs/2107.03374).
`pass@1 = c/n` es éxito esperado con muestreo; no significa greedy ni solo el
primer intento. `pass@n` equivale a la fracción de tareas con al menos un acierto.
Los textos repetidos sí cuentan como intentos en estas métricas.

Se incluyen solamente tareas con los n intentos planificados completos. Cada
columna lleva su denominador (`tasks@k`); también se muestra completadas/planificadas
y `provisional` si faltan tareas. Las corridas con distintas configuraciones o
identificadores se mantienen separadas. Duplicar un archivo de resultados al
agregarlo es un error, no aumenta artificialmente la muestra.

El pipeline genera `metrics.json`, `metrics.md`, `audit.json` y
`pipeline-status.json`. La auditoría vuelve a ejecutar reward sobre candidatos,
SFT, ambos miembros de cada par DPO y muestras de baseline; comprueba además que
los modelos se evaluaron sobre el mismo conjunto. `complete` es relativo al
alcance seleccionado: un piloto completo sigue siendo un piloto.

Las señales `zero_success`, `ceiling` y `mixed_success` describen lo observado.
No prueban dificultad intrínseca. Para calibrar, ampliar eval, separar errores
matemáticos de formato/truncamiento y buscar niveles donde el 2B tenga éxitos
y fallos. El dataset evita duplicados exactos, pero no todas las equivalencias
por permutaciones/simetrías del Sudoku.

## Corrida completa verificada el 4 de noviembre de 2026

La corrida [challenge-full-20261103](results/challenge-full-20261103) terminó
con **5.040 respuestas**: 450 tareas de train × 8 candidatos del 9B y 90 tareas
de eval × 8 respuestas por modelo. La [auditoría de entrega](results/challenge-full-20261103/delivery-audit.json)
reverificó las recompensas, los pares DPO y los exports, sin duplicados de intentos
ni muestras faltantes. Test sigue reservado; no se entrenaron pesos.

| Modelo | pass@1 | pass@2 | pass@4 | pass@8 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-2B original | 6.94% | 9.33% | 12.30% | 15.56% |
| Qwen3.5-9B MLX 4-bit | 17.22% | 24.29% | 32.67% | 40.00% |

Son promedios macro sobre las mismas 90 tareas de eval, con 10 tareas por
tamaño/nivel. El 2B obtuvo 50 respuestas correctas de 720 y resolvió 14 tareas
al menos una vez; el 9B obtuvo 124 de 720 y resolvió 36 tareas. El detalle por
grupo está en [metrics.md](results/challenge-full-20261103/metrics.md).

El profesor 9B produjo 566 candidatos correctos de 3.600. Tras el filtro y la
deduplicación por tarea, quedaron **161 ejemplos SFT de 161 tareas** y **600 pares
DPO de 137 tareas**. No hubo éxitos en train 6x6 hard ni 9x9 hard; 9x9 medium
aportó solo un ejemplo SFT. Las respuestas aceptadas contienen el tablero final,
sin una explicación de razonamiento: se usó `--no-thinking --answer-prefix`.
Los pares son preferencias sintéticas del verificador, no etiquetas humanas.

La corrida terminó, pero la cobertura sigue limitada: 289/450 tareas de train
no produjeron una respuesta correcta y el 9B alcanzó el límite de tokens en
1.197/3.600 candidatos y 268/720 respuestas de eval. Tamaño y cuantización difieren
entre modelos; esta comparación no aísla solo el número de parámetros. La
[revisión de simetrías 4x4](results/analysis-20261104-002956/symmetry-audit.json)
encontró equivalencias train/eval en 3/10 tareas de 4x4 hard (2/10 con SFT).
No hay duplicados exactos, pero deberán revisarse estas equivalencias antes de
atribuir a un ajuste futuro generalización a estructuras nuevas.

La [guía de entrega](results/challenge-full-20261103/ENTREGA_EQUIPO.md) reúne
cobertura, archivos principales, protocolo, limitaciones y ejemplos reales.
`source/` e `input/` dentro de la corrida conservan el código y los datos usados.
Los pilotos y el análisis intermedio permanecen separados de los resultados finales.

## Piloto ejecutado el 3 de noviembre de 2026

Artifacts en [results/challenge-prefill-pilot-20261103](results/challenge-prefill-pilot-20261103).
Se usaron 9 tareas de train y otras 9 de eval (una por tamaño/nivel), con 4
intentos por tarea. Es un piloto; cada celda tamaño/nivel contiene una sola tarea.

| Modelo | pass@1 | pass@2 | pass@4 |
| --- | ---: | ---: | ---: |
| Qwen3.5-2B original | 5.56% | 11.11% | 22.22% |
| Qwen3.5-9B MLX 4-bit | 22.22% | 31.48% | 44.44% |

Estos promedios macro usan las 9 tareas completas de eval. El detalle por tamaño
y nivel está en [metrics.md](results/challenge-prefill-pilot-20261103/metrics.md),
y los datos de la auditoría en [audit.json](results/challenge-prefill-pilot-20261103/audit.json).
El 2B tuvo un acierto de cuatro intentos en 4x4 fácil y otro en 6x6 fácil;
las otras siete tareas tuvieron cero aciertos. Es evidencia inicial de señal
para esas dos tareas, insuficiente para generalizar a todo un nivel.

El profesor generó 36 candidatos de train, de los cuales 3 fueron aceptados:
quedaron [2 trazas SFT distintas](results/challenge-prefill-pilot-20261103/traces-9b/sft.jsonl)
y [5 pares DPO](results/challenge-prefill-pilot-20261103/traces-9b/dpo.jsonl).
Las trazas SFT corresponden solo a 4x4 fácil y medio; falta ampliar volumen y
cobertura. La auditoría revalidó todas las recompensas y exports. La reanudación
del pipeline terminado se comprobó sin cargar modelos ni añadir candidatos.

Los primeros ensayos sin prefill están conservados como corridas incompletas en
`results/challenge-pilot-20261103` y `results/challenge-final-pilot-20261103`.
Se interrumpieron al observar explicaciones que agotaban el presupuesto. No forman
parte de esta línea base ni se combinaron sus muestras con el protocolo final.

## Progreso en terminal

`progress.py` muestra barras, porcentajes y contadores leyendo los checkpoints;
no importa modelos. Sus controles de pausa/reanudación operan sobre el lanzador
y los procesos de la corrida. La ventana es independiente de la
corrida: cerrarla o presionar `q` no detiene la generación.

```sh
python Laboratorio/Qwen3.5/progress.py --run results/challenge-full-20261103
```

Se actualiza cada dos segundos. Teclas: `1` resumen, `2` SFT/DPO por tamaño y
nivel, `3` pass@1/2/4/8 por modelo/tamaño/nivel, `4` log; flechas para desplazar
vistas largas. `--once` imprime una instantánea sin pantalla interactiva.

**`P` pausa y `R` reanuda.** `P` solicita una pausa persistente: espera el siguiente
checkpoint de una respuesta, detiene la inferencia y libera la memoria del modelo.
Esperar a que aparezca **PAUSED** antes de apagar el Mac. La pausa se conserva en
`control.json`, incluso tras reiniciar; el seguimiento automático y el lanzador
respetan esa decisión. `R` reanuda desde el siguiente intento pendiente con los mismos
datos, parámetros y semillas. Si la pausa todavía está esperando una respuesta,
`R` cancela la solicitud. Una vez que está deteniendo procesos, esperar a PAUSED.

La unidad de recuperación es una **respuesta completa**, no un token. El controlador
externo puede interrumpir el inicio del intento siguiente; ese intento aún no guardado
se repite con su semilla original. Nunca se repiten ni sobrescriben respuestas guardadas.
Cerrar la ventana conserva el estado actual de ejecución o pausa.

También se puede controlar desde otra terminal, en la raíz del repositorio:

```sh
python Laboratorio/Qwen3.5/run_control.py pause
python Laboratorio/Qwen3.5/run_control.py status
python Laboratorio/Qwen3.5/run_control.py resume
```

Los tres comandos aceptan `--run DIRECTORIO`. El controlador es externo al código
congelado de inferencia y no modifica el protocolo experimental. Verifica PID y
fecha de creación, detecta hijos huérfanos y rechaza lanzar una segunda inferencia.
Una pausa intencional bloquea `launch.py --resume`; usar `R` o el comando anterior
para indicar explícitamente que se desea continuar.

Los 3.600 candidatos y las 1.440 respuestas de evaluación son metas; las barras
cuentan intentos efectivamente guardados. Las 72 métricas son cuatro valores de
pass@k para cada uno de los 18 grupos modelo/tamaño/nivel. Se distinguen valores
pendientes (`--`), provisionales y finales; un 0% calculado no significa que
falte la métrica. SFT cuenta respuestas correctas distintas; DPO cuenta pares
verificados, por lo que sus cantidades no tienen una meta garantizada.

El lanzador para la corrida actual es
`results/challenge-full-20261103/Live Progress.command` (doble clic en Finder).

## Comparador exploratorio anterior

`compare_models.py` sigue disponible para los experimentos anteriores sobre
puzzles 9x9 resolubles por candidatos únicos y grupos de vacíos. Solo mide un
intento por problema; usar `baseline.py` para pass@k y los splits del Desafío.
Acepta `--resume DIRECTORIO` y recupera la configuración del manifiesto.
No mezclar sus resultados con la línea base estratificada del pipeline.

## Verificación y pendientes

```sh
python -m unittest discover -s Laboratorio/Qwen3.5 -p 'test_*.py'
```

Las pruebas no descargan pesos. Las del backend MLX requieren las dependencias
MLX de Mac; el resto prueba muestreo, estimador, reanudación y verificación con
respuestas controladas. Ver [TODO.md](TODO.md) para distinguir implementación,
datos efectivamente generados y evidencia experimental pendiente.
