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