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

`model.generate(case.prompt)` devuelve texto; `model.generate_with_stats(...)`
devuelve también los tokens y el motivo de término. Ambas aceptan las opciones
por llamada `max_tokens` y `enable_thinking`, que reemplazan los valores de la
configuración. `timed(funcion, argumentos...)` devuelve `(resultado, segundos)`
y `evaluate(resultado.text, case)` verifica la respuesta; para el texto de
`generate`, pasa directamente ese texto a `evaluate`.
