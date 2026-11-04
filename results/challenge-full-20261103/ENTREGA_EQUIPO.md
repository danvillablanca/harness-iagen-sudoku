# Entrega al equipo — Desafío Sudoku

Corrida completa, auditada el 4 de noviembre de 2026. **Se generaron datos y se
evaluaron modelos; no se entrenaron pesos.** Hay 5.040 respuestas guardadas y
verificadas, sin intentos duplicados ni muestras pendientes del alcance acordado.

## Abrir primero

1. Este documento: resumen y limitaciones.
2. `results/challenge-full-20261103/metrics.md`: pass@1/2/4/8 por tamaño y nivel.
3. `results/challenge-full-20261103/traces-9b/sft.jsonl`: **161 ejemplos SFT**.
4. `results/challenge-full-20261103/traces-9b/dpo.jsonl`: **600 pares DPO**.
5. `results/challenge-full-20261103/delivery-examples.json`: ejemplos reales de
   SFT, un par DPO y aciertos/fallos de ambos modelos, con sus identificadores.

Las rutas de esta guía son relativas a la raíz del paquete descomprimido.
Los JSONL contienen un registro JSON por línea; se pueden leer sin cargar modelos.

## Qué obtuvimos

El profesor Qwen3.5-9B generó 8 candidatos para cada una de las 450 tareas de
train: **3.600 candidatos, 566 correctos (15,72%)**. El filtro y la deduplicación
por tarea dejan **161 SFT de 161 tareas** (35,78% de cobertura) y **600 DPO de
137 tareas**. Los aciertos repetidos no se convierten en nuevas tareas aprendibles.
289 tareas no tuvieron aciertos; 24 tuvieron 8/8 aciertos y no aportan pares DPO.

Cada modelo de baseline respondió 8 veces a las mismas 90 tareas de eval:
**720 respuestas del 2B y 720 del 9B**, 10 tareas por tamaño/nivel.

| Modelo | pass@1 | pass@2 | pass@4 | pass@8 | Respuestas correctas | Tareas con algún acierto |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-2B original | 6.94% | 9.33% | 12.30% | 15.56% | 50/720 | 14/90 |
| Qwen3.5-9B MLX 4-bit | 17.22% | 24.29% | 32.67% | 40.00% | 124/720 | 36/90 |

Promedios macro sobre las 90 tareas (equivalentes al promedio de los nueve grupos
porque todos tienen 10 tareas). Para cada tarea, `pass@k = 1 - C(8-c,k)/C(8,k)`,
donde `c` es su número de aciertos. `pass@1` mide éxito con muestreo, no greedy;
`pass@8` es la fracción de tareas con al menos un acierto. No se cuenta una tarea
ocho veces como si fueran ocho problemas independientes.

El 9B supera al 2B bajo este protocolo, pero todavía falla en buena parte de las
tareas. No hay resultados antes/después de SFT o DPO: solo datasets destinados
a un posible entrenamiento posterior. Los resultados no muestran que un modelo
haya mejorado durante la corrida.

## Cobertura de los datos para SFT/DPO

Cada grupo tuvo 50 tareas de train y 400 candidatos.

| Grupo | Candidatos correctos | Tareas con SFT | Pares DPO |
| --- | ---: | ---: | ---: |
| 4x4/easy | 251/400 | 47/50 | 79 |
| 4x4/medium | 121/400 | 33/50 | 126 |
| 4x4/hard | 43/400 | 25/50 | 147 |
| 6x6/easy | 71/400 | 24/50 | 94 |
| 6x6/medium | 29/400 | 14/50 | 73 |
| 6x6/hard | 0/400 | 0/50 | 0 |
| 9x9/easy | 50/400 | 17/50 | 74 |
| 9x9/medium | 1/400 | 1/50 | 7 |
| 9x9/hard | 0/400 | 0/50 | 0 |

No hay ejemplos positivos de 6x6 hard ni de 9x9 hard. En 9x9 medium hay solo uno.
105/161 ejemplos SFT son 4x4; 38 son 6x6 y 18 son 9x9. Por tanto, la corrida
está completa y el dataset sigue siendo pequeño y desequilibrado.

## Qué significa una “traza” aquí

Se usó `thinking=false` y prefijo `<answer>\n[`. Los ejemplos SFT aceptados
contienen **el tablero final, sin razonamiento paso a paso**. No presentar este
paquete como un corpus de cadenas de razonamiento. Si se necesita supervisar
explícitamente el procedimiento, hay que diseñar una nueva corrida con ese objetivo.

`sft.jsonl` incluye `prompt`, `completion`, `messages`, `task_id` y `candidate_id`.
`dpo.jsonl` incluye `prompt`, `chosen`, `rejected`, `task_id`, `chosen_id` y
`rejected_id`. Cada par usa respuestas de la misma tarea, con `chosen=1` y
`rejected=0` según el verificador. Son preferencias sintéticas, no etiquetas humanas.
La recompensa valida formato, pistas y reglas del tablero; no certifica un razonamiento.

## Protocolo y límites de interpretación

- Modelo profesor y baseline mayor: `mlx-community/Qwen3.5-9B-4bit`, revisión
  `8b2b98c00a6b4d291155e4890773ca8f769aee53`.
- Baseline pequeño: `Qwen/Qwen3.5-2B` original, revisión
  `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- MLX, temperatura 0,7, top-p 0,8, top-k 20, penalización de presencia 1,5,
  semilla base 42 y semillas por tarea/intento; 1.024 tokens máximos de salida,
  2.048 de entrada; sin thinking, con prefill; no few-shot en esta corrida.
- Se alcanzó el límite de tokens en 1.197/3.600 candidatos del profesor (33,25%),
  268/720 respuestas de eval del 9B (37,22%) y 5/720 del 2B (0,69%).
  La recompensa también exige el formato; los ceros mezclan distintos fallos.
- Solo hay 10 tareas de eval por grupo. Las tasas observadas no son una
  calibración definitiva de dificultad. Los intentos de una tarea no son tareas independientes.
- Los modelos difieren tanto en tamaño como en precisión de pesos; no se aísla
  causalmente el efecto de tener más parámetros.
- La auditoría excluye duplicados exactos entre splits. La revisión adicional de
  simetrías 4x4 detectó 3/10 tareas hard de eval equivalentes a train, 2/10 a SFT.
  Esto limita futuras afirmaciones de generalización tras entrenar; no demuestra
  memorización de estos baselines sin ajuste. Ver `symmetry-audit.json` dentro de
  `results/analysis-20261104-002956/`. No se modificó el dataset congelado.
- **Test (90 tareas) sigue reservado**: está incluido para reproducibilidad,
  sin inferencia de modelos ni uso para generar SFT/DPO. La auditoría solo valida
  integridad, referencias y ausencia de duplicados exactos.

## Archivos y procedencia

| Ruta | Contenido |
| --- | --- |
| `results/challenge-full-20261103/` | Corrida final completa: fuente principal de resultados |
| `…/traces-9b/candidates.jsonl` | Las 3.600 respuestas crudas, aciertos y fallos, con semilla, tiempos y diagnóstico |
| `…/traces-9b/sft.jsonl`, `dpo.jsonl` | Exports filtrados y verificados |
| `…/baseline/*-nothinking.jsonl` | 720 respuestas por modelo, con metadata de cada intento |
| `…/input/` | Copia congelada y autoritativa de train/eval/test y manifest |
| `…/source/` | Código exacto de generación, evaluación y auditoría usado en la corrida |
| `…/snapshot.json`, `*/manifest.json` | Hashes, revisiones, parámetros y procedencia |
| `…/audit.json`, `delivery-audit.json` | Auditoría original y nueva comprobación de entrega |
| `…/delivery-summary.json` | Resumen final legible por programas, con promedios macro |
| `…/run.log` | Registro de ejecución completo |
| `Laboratorio/Qwen3.5/` | Código de trabajo, pruebas y herramientas de progreso/pausa |
| `data/sudoku-final/` | Dataset del protocolo final; `data/sudoku/` conserva el prompt anterior |
| `Colab_Sudoku.ipynb` | Cuaderno del proyecto; no es el registro de la corrida MLX del Mac |
| `Documentos Curso/` | Enunciado y PDFs de referencia del proyecto |
| `README.md`, `TODO.md` | Uso general del proyecto y pendientes |

Los otros directorios de `results/` conservan pilotos y ensayos anteriores;
**no sumar sus respuestas a las 5.040 finales**. En particular,
`results/analysis-20261104-002956/` es un análisis intermedio con 3.237 candidatos,
incluidas sus figuras y scripts; sus porcentajes no sustituyen los finales.
Los nombres históricos que mencionan 27B no forman parte de la comparación final.

No se incluyen pesos, cachés de modelos, entorno virtual, historial `.git`, locks,
descargas ni el informe formal. Los manifests y lanzadores originales conservan
rutas absolutas del Mac como procedencia. `launch.py`, `Live Progress.command` y
los scripts de análisis intermedio no son lanzadores portables; no ejecutarlos
para leer esta entrega. Los PID/estados de control son históricos: el estado final
autoritativo es `pipeline-status.json = complete`.

## Comprobar el ZIP sin modelos ni GPU

Desde la carpeta raíz descomprimida, con Python 3.10 o posterior (comprobado con
Python 3.13.12):

```sh
python3 -B VERIFICAR.py
```

Este script comprueba los SHA-256 del paquete, los hashes del código congelado,
las 3.600/720/720 muestras, los índices únicos por tarea y la ausencia de muestras
de test. Después ejecuta la auditoría congelada, reevalúa todas las recompensas,
verifica SFT y ambos miembros de DPO y contrasta las métricas. Usa solo la biblioteca
estándar y archivos locales; no instala dependencias, no descarga pesos ni hace inferencia.

Para ejecutar nuevas inferencias, consultar README y requirements. Crear otra
carpeta de resultados; no sobrescribir esta evidencia ni reutilizar estados de
lanzamiento del Mac. MLX requiere Apple Silicon; el entorno de inferencia original
está registrado en los manifests. La reproducibilidad numérica entre hardware
o versiones diferentes no está garantizada.
