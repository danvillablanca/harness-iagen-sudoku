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
El prompt del puzzle pide entonces el mismo procedimiento paso a paso, y el
parser acepta el razonamiento previo al bloque `<answer>`.

Al terminar, el harness imprime tiempos, tokens generados, motivo de término,
número de shots, el puzzle, la solución esperada, la solución del modelo, la
entrada completa al modelo (shots incluidos) y su salida completa.

```sh
python Laboratorio/Qwen3.5/main.py --no-thinking --empty-cells 20 --seed 1 --shots 2
```

`model.generate(case.prompt)` devuelve texto; `model.generate_with_stats(...)`
devuelve también los tokens y el motivo de término. Ambas aceptan las opciones
por llamada `max_tokens`, `enable_thinking` y `shots` (pares usuario/asistente
antes del prompt), que reemplazan los valores de la configuración. `timed(funcion, argumentos...)` devuelve `(resultado, segundos)`
y `evaluate(resultado.text, case)` verifica la respuesta; para el texto de
`generate`, pasa directamente ese texto a `evaluate`.
