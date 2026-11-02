# TODO

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
