# V5-B2 — Comparing schematic and solid joint representations

This notebook comparison explores two Three.js representations of
revolute (R) and prismatic (P) joints:

- **schematic**: arcs, arrows, and guide lines representing joint motion.
  This representation is also available in Matplotlib.
- **solid**: a blue cylinder for R and an orange rectangular prism for P,
  aligned with `JointData.axis` and centered on `JointData.origin`.
  This is the default appearance within the `kinematic` representation.

The combined `hybrid` representation was discarded after visual inspection
because it added excessive visual information to the scene.

The solids are **kinematic motion symbols**, not models of the robot's
physical actuators or hardware. In particular, a prismatic symbol does not
slide independently along its DH displacement parameter `d_i`. It indicates
the location and direction of the **joint motion axis**, while the evaluated
DH geometry reflects the robot's current configuration.

All dimensions are derived from `VisualizationLayout` and `joint_scale`.
The solids were reduced by 20% relative to the first prototype and use
joint-type colors: R `#0072B2` and P `#D55E00`. The colors can be changed
independently with `revolute_color` and `prismatic_color`. The nominal
radii and lengths scale consistently with the robot's linear units,
including meters and millimeters. Three.js objects are reused between
animation frames; their positions and orientations are updated in place.

**Intentional limitation:** `solid` is only supported in Three.js.
Matplotlib requires `joint_appearance="schematic"` when
`joint_representation="kinematic"` is requested.

## Minimal notebook / Colab comparison

Use Moro 0.6.0 (or the `release/0.6.0` candidate before publication)
and restart the notebook kernel after installation or updating:

```python
from sympy import symbols, pi
from moro.core import Robot
from moro.visualization import RobotVisualizer, VisualizationStyle
from IPython.display import display

q1, q2, d3 = symbols("q1 q2 d3")

robots = {
    "Planar RR": (
        Robot((0.7, 0, 0, q1), (0.5, 0, 0, q2)),
        {q1: 0.4, q2: -0.6},
    ),
    "Spatial RP": (
        Robot((0.7, pi/2, 0.2, q1, "r"), (0.4, 0, d3, 0, "p")),
        {q1: 0.4, d3: 0.35},
    ),
    "Spatial RRP": (
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

To compare the two appearances, try `joint_scale=0.7`, `1.0`, and
`1.4`. Hiding coordinate frames (`show_frames=False`) and the base
(`show_base=False`) independently can also help assess readability.

## Reference prismatic-joint animation

```python
from numpy import linspace

robot, _ = robots["Spatial RP"]
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

The expected behavior is that the P motion symbol remains aligned with
its joint axis, while the evaluated DH segment associated with the
second joint's displacement changes (`d_2`, represented by the Python
symbol `d3` in this example).

## Evaluation criteria and final V5-B2 decision

Compare R/P legibility, oblique-axis orientation, visibility of DH
segments, overlap near the base, zoom behavior, camera interactions,
and visual consistency across meters and millimeters.

**V5-B2 decision:** `solid` is the default appearance when the
`kinematic` representation is selected in Three.js; `schematic`
remains available as an alternative. Matplotlib continues to support
schematic kinematic joints, but not solid R/P symbols.

**Subsequent V5-C2 change:** Moro 0.6.0 uses
`joint_representation="auto"` as its general default. Three.js
resolves `auto` to solid kinematic joints when complete joint metadata
is available; Matplotlib and legacy scenes without that metadata
retain simple spherical markers. Users can explicitly request
`joint_representation="simple"` in either backend.

The possibility of implementing solid R/P symbols in Matplotlib
remains a separate future consideration.
