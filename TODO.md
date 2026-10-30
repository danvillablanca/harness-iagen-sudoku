# TODO

## Validar los steps en el parser

Con `--shots > 0`, `parse_answer(..., allow_steps=True)` (`Laboratorio/Qwen3.5/response_parser.py`)
solo comprueba que el texto **termine** en un bloque `<answer>...</answer>` bien formado
(`text.endswith(match.group(0))`). El texto anterior no se valida.

Por ahora es intencional, para mantenerlo simple. Una respuesta sin líneas `Step k:`, o con texto
libre antes del bloque, igual obtiene `Required answer format = True`, aunque no siga el
procedimiento paso a paso de los shots.

Pendiente: validar el texto previo al `<answer>` cuando `allow_steps=True`:

- que existan líneas `Step k: rXcY = v`
- que `k` sea consecutivo desde 1
- que cada paso llene exactamente una celda
- que haya una grilla después de cada paso

Opcional: registrar la cantidad de pasos como dato aparte.

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
