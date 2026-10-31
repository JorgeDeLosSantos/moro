# V5-B2 — Experimento visual de juntas: A/B/C

Este experimento permite comparar tres alternativas en Three.js **sin cambiar el
modo de visualización predeterminado de Moro**:

- **A · schematic**: arcos/flechas y guías en forma de líneas. Funciona también
  en Matplotlib, y sigue siendo el modo inicial de V5-B2.
- **B · solid**: cilindro para R, prisma rectangular para P, alineados con
  `JointData.axis` y centrados en `JointData.origin`.
- **C · hybrid**: los mismos sólidos, semitransparentes, junto a las líneas de
  A para indicar el sentido de movimiento.

Los sólidos son **símbolos de movimiento**; no describen los componentes
físicos del robot. Los elementos prismáticos no se deslizan de manera
independiente a lo largo del parámetro DH `d_i`; muestran dónde está el
**eje de movimiento**, mientras la geometría DH sigue reflejando la
configuración evaluada.

Todas las dimensiones proceden de `VisualizationLayout` y
`joint_scale`. El radio y la longitud de los volúmenes son independientes
de que las distancias estén expresadas en m o mm. Los objetos de Three.js
se conservan entre fotogramas; se actualizan su origen y dirección.

**Limitación deliberada:** `solid` y `hybrid` solo están disponibles
en Three.js. Matplotlib solicita explícitamente `schematic` para
evitar presentar variantes visuales distintas bajo el mismo nombre.

## Comparador mínimo para notebook / Colab

Instala la rama `feat/0.6.0-joint-symbols` y reinicia el kernel. Ejecuta:

```python
from sympy import symbols, pi
from moro.core import Robot
from moro.visualization import RobotVisualizer, VisualizationStyle
from IPython.display import display

q1, q2, d3 = symbols("q1 q2 d3")

robots = {
    "RR planar": (
        Robot((0.7, 0, 0, q1), (0.5, 0, 0, q2)),
        {q1: 0.4, q2: -0.6},
    ),
    "RP espacial": (
        Robot((0.7, pi/2, 0.2, q1, "r"), (0.4, 0, d3, 0, "p")),
        {q1: 0.4, d3: 0.35},
    ),
    "RRP espacial": (
        Robot(
            (0.7, pi/2, 0.2, q1, "r"),
            (0.4, -pi/3, 0.1, q2, "r"),
            (0.3, 0, d3, 0, "p"),
        ),
        {q1: 0.4, q2: -0.3, d3: 0.35},
    ),
}

for name, (robot, q) in robots.items():
    print(name)
    for appearance in ("schematic", "solid", "hybrid"):
        print("   ", appearance)
        style = VisualizationStyle(
            joint_representation="kinematic",
            joint_appearance=appearance,
            joint_scale=1.0,
        )
        display(
            RobotVisualizer(robot).plot(q, backend="threejs", style=style)
        )
```

Para comparar la apariencia, intenta con
`joint_scale=0.7`, `1.0` y `1.4`. También es útil ocultar los frames
(`show_frames=False`) y la base (`show_base=False`) de manera independiente.

## Animación prismática de referencia

```python
from numpy import linspace

robot, _ = robots["RP espacial"]
configurations = [
    {q1: 0.5, d3: float(d)}
    for d in linspace(-0.3, 0.5, 25)
]

style = VisualizationStyle(
    joint_representation="kinematic",
    joint_appearance="hybrid",
)
display(
    RobotVisualizer(robot).animate(
        configurations, backend="threejs", style=style
    )
)
```

El resultado esperado es que el símbolo P permanezca sobre su eje,
mientras cambia el segmento DH asociado al desplazamiento `d_3`.

## Pautas para elegir variante

Considera legibilidad de R/P, orientación de ejes oblicuos, visibilidad
de los segmentos DH, superposición en la base, comportamiento con zoom,
comportamiento de los controles y consistencia entre m y mm.

No se modifica automáticamente la representación predeterminada.
La elección definitiva y la eventual adaptación de sólidos a Matplotlib
se decidirán después de la inspección visual.
