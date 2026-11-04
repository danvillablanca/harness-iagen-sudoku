# TODO

## Desafío: implementación

- [x] Harness común: prompt, parser, verificador y `reward(prompt, completion)`.
- [x] Integrar `origin/baseline` de Lilly conservando el generador y el trabajo local.
- [x] Dataset reproducible 4x4/6x6/9x9, easy/medium/hard, train/eval/test.
- [x] Variante de prompt final con los mismos puzzles, sin sobrescribir la original.
- [x] Muestreo positivo, semillas por intento, top-p/top-k y penalización de presencia.
- [x] Profesor 9B: candidatos, filtro real con reward, SFT y pares DPO de la misma tarea.
- [x] Reanudación por intento, recuperación de escritura final incompleta y exports reconstruibles.
- [x] Pausa persistente con P y reanudación con R en terminal; seguimiento respeta control.json.
- [x] Baseline 2B/9B en tareas idénticas de eval; pass@1/pass@k por tamaño y nivel.
- [x] Impedir mezcla de configuraciones, duplicación de muestras y uso de train como baseline.
- [x] Métricas con denominadores, cobertura y estado provisional en corridas parciales.
- [x] Pipeline `challenge.py` con modelos secuenciales y auditoría de los artifacts.
- [x] Pruebas adversariales y regresiones de muestreo, persistencia, prefill y estimador.

## Evidencia y alcance

Corrida completa iniciada el 2026-11-03 y finalizada; entrega auditada el 2026-11-04:
`results/challenge-full-20261103/`. Alcance: 450 train × 8 candidatos del 9B;
90 eval × 8 respuestas para cada modelo (2B y 9B), 5.040 muestras en total.
Código y datos congelados; revisiones de pesos fijadas sin copiar pesos.
Estado en `pipeline-status.json`, progreso en `run.log`; pausa/reanudación mediante
P/R en `progress.py` o `run_control.py pause|resume`. La pausa queda en `control.json`
y bloquea reinicios automáticos. `launch.py --resume` recupera fallos espontáneos
solo cuando no hay una pausa intencional. El seguimiento automático de este chat fue
cancelado a petición del usuario el 2026-11-03; ya no hay recuperación ni avisos programados.
`pipeline-status.json` está en `complete`; `delivery-audit.json` confirma todas las
muestras y recompensas. Resultado: 161 SFT, 600 DPO; 2B pass@1/8 = 6.94%/15.56%,
9B pass@1/8 = 17.22%/40.00%. No se entrenaron pesos ni se evaluaron modelos en test.


- [x] Generar y verificar 450 train / 90 eval / 90 test en `data/sudoku-final`.
  Son los mismos puzzles que `data/sudoku`, con otra instrucción de salida.
- [x] Finalizar y auditar el piloto real de 9 tareas por split y 4 muestras por tarea:
  `results/challenge-prefill-pilot-20261103/`. 36 candidatos del 9B, 2 SFT
  distintas y 5 pares DPO; 36 respuestas de eval por modelo. Reanudación completa
  comprobada sin cargar pesos ni duplicar muestras.
- [x] Completar los 3.600 candidatos de las 450 tareas de train y auditar los exports:
  566 candidatos correctos, 161 SFT y 600 DPO. Cobertura SFT: 161/450 tareas.
- [x] Ampliar eval a las mismas 90 tareas × 8 respuestas por modelo; 720 muestras cada uno,
  pass@1/2/4/8 finales en los 18 grupos modelo/tamaño/nivel.
- [ ] Mejorar cobertura y calibración en una corrida nueva: cero éxitos del profesor en
  6x6 hard y 9x9 hard; solo un ejemplo SFT de 9x9 medium. Revisar formato/truncamiento.
- [ ] Fijar protocolo y hacer la evaluación final en test (hasta entonces, reservado).
- [ ] Revisar equivalencias por simetrías/permutaciones: se excluyen duplicados exactos,
  no todas las clases isomorfas del Sudoku. La revisión 4x4 detectó 3/10 tareas de eval
  hard equivalentes a train y 2/10 a SFT; pendiente resolver para futuros ajustes.
- [ ] Versionar/publicar código y artifacts seleccionados; los pesos quedan en caché externa.

Los resultados con prefill de `<answer>\n[` son de esa configuración del harness;
no deben mezclarse con las pruebas anteriores sin prefill o con thinking.
El prefijo no introduce valores ni soluciones. El verificador evalúa el tablero
final y el formato; no certifica el razonamiento paso a paso. No se entrena un
modelo SFT/DPO/RL en esta etapa del Desafío.

## Crédito parcial y dificultad

La recompensa actual es binaria. Para añadir crédito parcial, definir y evaluar
una medida que no penalice soluciones alternativas válidas. Una opción es contar
coincidencias en las casillas originalmente vacías contra la solución válida más
cercana, con todas las coincidencias pertenecientes a una misma solución.
Hace falta resolver o acotar esa optimización y declarar qué ocurre si no puede
completarse; no devolver un porcentaje aproximado como si fuera exacto.

Si se demuestra unicidad para una instancia, la comparación con su única solución
es directa. Mantener separado el éxito completo (`passed`) del puntaje parcial.
Definir los pesos configurables y su normalización para conservar el rango [0, 1].

## Análisis opcional de los pasos

El contrato ahora permite explicaciones y pasos, pero solo evalúa el tablero
del único bloque final. No se exige ni se valida la secuencia `Step k:`.
Si se estudia la calidad del procedimiento, registrar esa evaluación por separado
sin convertirla implícitamente en requisito de formato.

## Semillas y tope de reintentos en `make_shots`

`make_shots` (`Laboratorio/Qwen3.5/prompts.py`) usa `shot_seed = seed * 1000 + attempt`:

- Si `attempt` supera 999, el rango de un `--shot-seed` invade el del siguiente (`--shot-seed 0`
  con `attempt = 1001` repite `--shot-seed 1` con `attempt = 1`) y dos corridas distintas podrían
  usar shots idénticos.
- `while len(shots) < n_shots` no tiene tope: si `continue` (puzzle igual al de prueba) se activara
  siempre, el bucle sería infinito.

Ambos casos son poco probables hoy (`attempt` solo pasa de `n_shots` cuando se descarta un puzzle).
Es el pendiente menos urgente.

Pendiente:

- derivar la semilla sin solape, p. ej. `random.Random((seed, attempt))` o `seed * 10**6 + attempt`
- poner un tope (`for attempt in range(MAX_ATTEMPTS)`) y lanzar un `ValueError` claro si no se
  consiguen los shots
