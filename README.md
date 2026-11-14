# harness-iagen-sudoku
Harness de Sudoku para el curso IIC3694-1 Inteligencia Artificial Generativa.

Hay una versión del harness para Google Colab en `Colab_Sudoku.ipynb`.

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
python Laboratorio/Qwen3.5/dataset.py --sizes 9 --train 200 --eval 50 --test 50 --seed 1
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

Tests (sin dependencias adicionales; hoy cubren `dataset.py` y `baseline.py`):

```sh
python -m unittest discover -s Laboratorio/Qwen3.5 -p "test_*.py"
```

## Línea base (pass@1 y pass@k)

`Laboratorio/Qwen3.5/baseline.py` genera `n` respuestas por tarea con
muestreo (en un solo batch con PyTorch), las puntúa con `reward` y guarda una
línea JSON por muestra en `results/<modelo>-<thinking|nothinking>.jsonl`.
Si se interrumpe, al repetir el mismo comando continúa desde donde quedó;
con otros parámetros de muestreo sobre el mismo archivo, se detiene con un error.

```sh
python Laboratorio/Qwen3.5/dataset.py --out data/sudoku
python Laboratorio/Qwen3.5/baseline.py run --tasks data/sudoku/eval.jsonl \
    --model-id Qwen/Qwen3.5-2B --no-thinking --per-level 5 --n 8
python Laboratorio/Qwen3.5/baseline.py summary results/*.jsonl --out results/summary.json
```

| Argumento de `run` | Descripción | Por defecto |
| --- | --- | --- |
| `--tasks FILE` | JSONL de `dataset.py` | requerido |
| `--model-id ID` | Modelo de Hugging Face | `Qwen/Qwen3.5-2B` |
| `--backend B` | `auto`, `mlx`, `cuda`, `mps` o `cpu` | `auto` |
| `--thinking` / `--no-thinking` | Modo de razonamiento | desactivado |
| `--n N` | Muestras por tarea | 8 |
| `--batch-size N` | Muestras generadas a la vez (bajar si falta memoria) | 8 |
| `--max-output-tokens N` | Límite de tokens por muestra | 2048 |
| `--temperature`, `--top-p` | Muestreo | 0.7 / 0.8 sin thinking; 0.6 / 0.95 con thinking |
| `--per-level N` | Primeras N tareas por tamaño y nivel | todas |
| `--sizes N...` | Filtrar tamaños de tablero | todos |
| `--seed N` | Semilla del muestreo | 0 |
| `--out DIR` | Carpeta de resultados | `results` |

`summary` acepta `--k N...` (valores de k para pass@k; por defecto `1 2 4 8`) y
`--out FILE` (salida JSON opcional). Imprime una tabla Markdown por modelo, modo, tamaño y nivel con
pass@k (estimador insesgado de Chen et al., 2021: `1 - C(n-c, k) / C(n, k)`,
promediado sobre tareas), el porcentaje de respuestas cortadas por el límite de
tokens y los tokens promedio. pass@k solo se calcula para tareas con al menos
`k` muestras.
