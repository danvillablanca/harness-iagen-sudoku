# Propuesta de reward con crédito parcial

## Fórmula

```
reward = format · ( w_fill · fill · clues_frac  +  w_solve · solved )
```

- `w_fill + w_solve = 1`, ambos ≥ 0 (propuesta: 0.5 / 0.5).
- Todos los factores están en [0, 1], así que el reward queda en [0, 1].
- Una solución correcta da 1.0.
- `passed` (éxito completo) sigue siendo binario y separado del puntaje parcial.

## Componentes

| Componente | Rango | Qué mide |
|---|---|---|
| `format` | {0, 1} | Un único bloque final `<answer>`, sin texto después, con una grilla NxN de enteros. Solo estructura, no contenido. |
| `clues_frac` | [0, 1] | Pistas del input que siguen idénticas en la salida / total de pistas. |
| `fill` | [0, 1] | Casillas originalmente vacías cuyo valor coincide con la solución única / total de vacías. |
| `solved` | {0, 1} | Grilla válida (filas, columnas, cajas) con todas las pistas intactas. Acepta cualquier solución válida. |

**Por qué `format` está separado de `clues_frac`**: un 0 en `format` significa que
el modelo no cumple el contrato de salida; un `clues_frac` bajo significa que
copia mal el input. Son fallas distintas.

**Por qué `format` multiplica todo**: con un término aditivo, el modelo podría
cobrar por el formato sin resolver nada. Si el formato falla, el reward es 0.

**Salidas con ceros o valores fuera de rango**: pasan `format`, porque el parser
solo exige enteros no negativos y el contenido se valida después. Ese 0 (o un
valor mayor que N) no coincide con la solución, así que `fill` lo cuenta como
error, igual que un valor equivocado: no hay incentivo a dejar ceros en lugar de
adivinar. `solved` sigue exigiendo que no quede ninguno. Se mantiene así para no
borrar el crédito parcial de las respuestas incompletas, aunque el prompt pida que
no queden ceros.

**Por qué `clues_frac` es continuo**: alterar una pista nunca sube el reward
(`fill` solo cuenta casillas vacías), así que no hay incentivo a hacerlo; el
castigo proporcional da gradiente. El rigor está en `solved`, que exige todas las
pistas intactas. Limitación: alterar 1 de 30 pistas resta solo ~3 %.

**Por qué `fill` compara con la solución única**: si el puzzle admitiera varias
soluciones y se comparara con una sola referencia, una respuesta correcta
distinta de la referencia recibiría `fill` bajo, y una inválida podría puntuar
más que una válida. Con solución única, toda respuesta correcta es la referencia.

## Unicidad

`reward` solo recibe el prompt, así que no puede confiar en una etiqueta del
generador: comprueba la unicidad con un solver (búsqueda con tope de 2 soluciones,
cacheada por puzzle).

- **Exactamente 1 solución**: `fill` se calcula contra ella.
- **2 o más soluciones**: sin crédito parcial, `reward = format · solved`.

Se prefiere este fallback binario a una medida aproximada (p. ej. contar casillas
sin conflicto) porque el float no puede indicar que el valor es aproximado: dos
significados en la misma escala serían indistinguibles. Nuestros datos son siempre
únicos, así que el fallback casi no se ejecuta.

## Ejemplos (9x9, 31 pistas, 50 vacías, pesos 0.5 / 0.5)

| Caso | `format` | `clues_frac` | `fill` | `solved` | reward |
|---|---|---|---|---|---|
| Solución correcta | 1 | 1.000 | 1.00 | 1 | 1.000 |
| Pistas intactas, 40 de 50 vacías correctas | 1 | 1.000 | 0.80 | 0 | 0.400 |
| Igual, con 1 pista alterada | 1 | 0.968 | 0.80 | 0 | 0.387 |
| Sin `<answer>` o con texto después | 0 | n/a | n/a | n/a | 0.000 |
| Puzzle no único, solución válida | 1 | 1.000 | n/a | 1 | 1.000 |
| Puzzle no único, solución inválida | 1 | 1.000 | n/a | 0 | 0.000 |

## Datos que respaldan la propuesta

Los generadores del repositorio producen puzzles de solución única, y el solver
es barato (medido con los `LEVELS` actuales):

- Todos los niveles de `dataset.py` son alcanzables con unicidad: máximo de
  vacías observado 12 (4x4), 27 (6x6) y 59 (9x9), frente a los 12, 24 y 50 que
  piden los niveles más difíciles.
- No se agotan los puzzles distintos: 3000 por nivel en 4x4 y 400 en 6x6 (topes
  de la medición) sin repetir.
- `count_solutions` en 9x9: mediana ≤ 0.0012 s y máximo 0.0068 s por puzzle.
  `sudoku.json` (60 vacías, fuera de `LEVELS`) tarda 0.72 s, por eso se cachea.
- `sudoku.json` y los puzzles de `make_case(empty_cells=...)` tienen solución
  única.

## Puntos para discutir

1. **Pesos** `w_fill` / `w_solve`: ¿0.5 / 0.5 o dar más peso a `solved`?
2. **Castigo de pistas alteradas**: ¿basta `clues_frac` lineal, o elevarlo a una
   potencia `k > 1`? Conviene decidirlo con datos de entrenamiento.
3. **Reward escaso al inicio**, si el modelo casi nunca acierta el formato. No se
   propone un término aditivo por formato hasta ver datos.
4. **Dificultad**: no se pondera dentro del reward. Si el algoritmo de
   entrenamiento normaliza por grupo, el peso por dificultad se cancela; si no, se
   puede lograr muestreando por nivel o ponderando al agregar métricas. Se retoma
   cuando se elija el algoritmo. Corregir `fill` por azar (`b = 1/N`) queda como
   opción si se observa que el piso distorsiona.
