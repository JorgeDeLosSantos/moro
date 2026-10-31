# V5-B2 — Comparación de juntas: esquemática y sólida

Esta comparación permite explorar dos alternativas en Three.js **sin cambiar
la representación general predeterminada de Moro**:

- **schematic**: arcos/flechas y guías en forma de líneas. Funciona también
  en Matplotlib.
- **solid**: cilindro azul para R, prisma rectangular naranja para P, alineados
  con `JointData.axis` y centrados en `JointData.origin`. Es la opción
  predeterminada **dentro del modo cinemático**, pero `simple` sigue siendo
  el modo predeterminado general.

La variante superpuesta `hybrid` se descartó después de la inspección visual:
añadía demasiada información a la escena.

Los sólidos son **símbolos de movimiento**; no describen los componentes
físicos del robot. Los elementos prismáticos no se deslizan de manera
independiente a lo largo del parámetro DH `d_i`; muestran dónde está el
**eje de movimiento**, mientras la geometría DH sigue reflejando la
configuración evaluada.

Todas las dimensiones proceden de `VisualizationLayout` y
`joint_scale`. Los sólidos son un 20 % menores que el primer prototipo y
se colorean según el tipo: R `#0072B2`, P `#D55E00`. Los colores pueden
configurarse mediante `revolute_color` y `prismatic_color`. El radio
y la longitud de los volúmenes son independientes
de que las distancias estén expresadas en m o mm. Los objetos de Three.js
se conservan entre fotogramas; se actualizan su origen y dirección.

**Limitación deliberada:** `solid` solo está disponible en Three.js.
Matplotlib requiere `joint_appearance="schematic"` cuando se solicita
`joint_representation="kinematic"`.

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
    for appearance in ("solid", "schematic"):
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
    joint_appearance="solid",
)
display(
    RobotVisualizer(robot).animate(
        configurations, backend="threejs", style=style
    )
)
```

El resultado esperado es que el símbolo P permanezca sobre su eje,
mientras cambia el segmento DH asociado al desplazamiento de la segunda junta (`d_2`, representado por el símbolo Python `d3` en este ejemplo).

## Pautas para elegir variante

Considera legibilidad de R/P, orientación de ejes oblicuos, visibilidad
de los segmentos DH, superposición en la base, comportamiento con zoom,
comportamiento de los controles y consistencia entre m y mm.

Se conserva `joint_representation="simple"` como predeterminado global.
En el modo cinemático, Three.js utiliza `solid` salvo indicación contraria;
la representación `schematic` se mantiene como alternativa. La eventual
adaptación de sólidos a Matplotlib se evaluará por separado.
