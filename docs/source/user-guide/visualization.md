# Visualization

`moro` provides visualization tools for inspecting robot configurations and animations.

The main interface is `RobotVisualizer`, which takes a `Robot` model and evaluates its symbolic kinematics at numerical configurations before passing the resulting scene to one of the available rendering backends.

Two backends are currently available:

* **Three.js** (default), for interactive browser-based visualization in
  Jupyter and Colab;
* **Matplotlib** (explicit), for standard Python figures and animations.

This section focuses on the recommended high-level workflow. Lower-level visualization objects such as `SceneData`, `FrameData`, and the backend classes are documented in the **API Reference**.

## Creating a visualizer

Start by defining a robot:

```python id="6noii4"
from moro import Robot
from moro.abc import q1, q2, l1, l2

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

Then create a visualizer:

```python id="a4b0s7"
from moro.visualization import RobotVisualizer

viz = RobotVisualizer(robot)
```

A `RobotVisualizer` is associated with a single `Robot` instance.

The same visualizer can then be reused for different configurations, backends, and animations.

## Plotting a robot configuration

A single robot configuration is displayed with:

```python id="wc2cxo"
viz.plot(...)
```

For example:

```python id="u1t5x2"
values = {
    l1: 1.0,
    l2: 0.8,
    q1: 0.5,
    q2: 0.8,
}

viz.plot(values)
```

The first argument is a dictionary that maps symbolic robot parameters to numerical values.

Because Three.js is the default backend, the previous call is equivalent to:

```python id="xlv49t"
viz.plot(
    values,
    backend="threejs",
)
```

The result is an `IPython.display.HTML` viewer. Code expecting
`fig, ax = ...` must request `backend="matplotlib"` explicitly.

## Providing numerical values

Visualization requires numerical frame transformations.

For a fully numerical robot model, only the joint variables may need to be supplied:

```python id="308duh"
robot = Robot(
    (1.0, 0, 0, q1, "r"),
    (0.8, 0, 0, q2, "r"),
)

values = {
    q1: 0.5,
    q2: 0.8,
}
```

For a symbolic robot model:

```python id="uyy3he"
robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

the geometric parameters must also be provided:

```python id="90a2fx"
values = {
    l1: 1.0,
    l2: 0.8,
    q1: 0.5,
    q2: 0.8,
}
```

Internally, `moro` evaluates the transformation of each robot frame with respect to the base frame and converts the resulting matrices to numerical scene data.

Therefore, all symbolic quantities required by the robot transformations must have numerical values before rendering.

## Choosing a backend

The backend is selected with:

```python id="dq7szp"
backend=...
```

The available values are:

```text id="1z8xnc"
"threejs"     # default
"matplotlib"  # explicit alternative
```

For example:

```python id="ekkuuw"
viz.plot(
    values,
    backend="matplotlib",
)
```

or:

```python id="gbhcta"
viz.plot(
    values,
    backend="threejs",
)
```

Both backends use the same evaluated robot geometry and the same `VisualizationStyle` interface, but their rendering capabilities differ.

## Matplotlib visualization

The Matplotlib backend creates a standard 3D Matplotlib figure.

Use:

```python id="5hi25s"
fig, ax = viz.plot(
    values,
    backend="matplotlib",
)
```

The result contains:

```text id="0dn5zq"
fig → Matplotlib Figure
ax  → 3D Matplotlib Axes
```

This makes it possible to continue customizing the plot using normal Matplotlib operations.

For example:

```python id="241h0y"
ax.set_title("Planar 2R robot")
```

The backend does not call `plt.show()` automatically, so display remains under user control.

In scripts, the figure can be shown with:

```python id="e3pjhv"
import matplotlib.pyplot as plt

plt.show()
```

### Figure size

The default figure size is:

```python id="4ta8qp"
figsize=(10, 8)
```

and can be changed with:

```python id="it0n71"
fig, ax = viz.plot(
    values,
    backend="matplotlib",
    figsize=(8, 6),
)
```

### Camera orientation

The initial camera orientation can be configured with:

```python id="j6nzvd"
view_init=(elevation, azimuth)
```

For example:

```python id="3spkw0"
fig, ax = viz.plot(
    values,
    backend="matplotlib",
    view_init=(40, 25),
)
```

This follows the usual Matplotlib 3D view convention.

## Interactive Three.js visualization

The Three.js backend produces an interactive view intended primarily for notebook environments.

Use:

```python id="abmqhd"
viz.plot(
    values,
    backend="threejs",
)
```

The result is returned as an `IPython.display.HTML` object and can be displayed directly in environments such as Jupyter Notebook or JupyterLab.

The browser viewer loads Three.js and OrbitControls from a CDN; the
browser needs network access to those resources. For standalone Python
scripts or offline rendering, explicitly choose Matplotlib.

The default rendering area is:

```python id="9u02rc"
width=800
height=600
```

and can be changed with:

```python id="dr0h9h"
viz.plot(
    values,
    backend="threejs",
    width=900,
    height=650,
)
```

### Preset views

The interactive viewer includes buttons for:

```text id="br7rsv"
Front
Right
Top
Isometric
```

These controls change the camera directly inside the viewer.

The initial view is isometric.

### Camera type

The viewer supports both:

```text id="lzlfbj"
Orthographic
Perspective
```

projection.

The orthographic camera is selected initially.

The camera type can be changed interactively without recomputing the robot model.

### Interactive navigation

The Three.js viewer also provides free camera navigation through orbit controls.

The user can rotate, zoom, and inspect the robot from arbitrary directions.

Use left-drag or a touchpad primary drag to orbit, right-drag to pan,
and the wheel or pinch gesture to zoom. On touchscreens, one finger orbits
and two fingers zoom/pan.

The **Fit View** button restores the scene center and fitting scale without
changing the current viewing orientation or perspective/orthographic
projection. Use **Isometric** to restore that camera orientation.

The preset view and camera controls belong to the Three.js interface itself; they are not arguments of `RobotVisualizer.plot()`.

## Default joint representation

By default, `VisualizationStyle(joint_representation="auto")` selects
joint representations according to the chosen backend:

* Three.js renders solid blue revolute cylinders and orange prismatic
  boxes when the scene contains evaluated R/P joint metadata.
* Matplotlib retains simple spherical joint markers.
* Manually assembled legacy scenes without joint metadata use simple
  spheres, including in Three.js.

This does not modify the caller's style object. To force spherical
markers, use `joint_representation="simple"`. To select schematic
joint motion lines in either backend:

```python
style = VisualizationStyle(
    joint_representation="kinematic",
    joint_appearance="schematic",
)
viz.plot(values, style=style)
```

The solid kinematic appearance currently requires Three.js. Matplotlib
raises an explanatory error if explicitly asked to render solid joints.

## Animation playback interval

Both backends accept `interval`, measured in milliseconds per animation
frame, with a default of 100 ms:

```python
viz.animate(configurations, interval=50)
```

For Three.js this is a positive finite number and controls the browser's
playback timer, not the physical time coordinates of a computed trajectory.

### Physical-time playback in Three.js (T-06)

The default Three.js backend can reproduce joint configurations at explicit
**physical timestamps**, including nonuniform intervals:

```python
# For a trajectory with matching time-major q and physical times t:
viz.animate(traj.q, times=traj.t, playback_speed=1.0)
```

`times` has one strictly increasing, finite real timestamp in **seconds**
per frame, with at least two samples. Its first value need not be zero:
Three.js uses the elapsed time relative to the first timestamp.
`playback_speed` is a strictly positive finite multiplier (1.0 = nominal
physical speed, 2.0 = twice as fast, 0.5 = half speed). No intermediate
joint configurations are interpolated inside the visualizer.

For example, a minimal one-joint animation that holds unevenly spaced
samples can be created as follows:

```python
from moro import Robot
from moro.abc import q1
from moro.visualization import RobotVisualizer

robot = Robot((1.0, 0, 0, q1, "r"))
viz = RobotVisualizer(robot)

viz.animate(
    [[0.0], [0.2], [0.5], [0.8], [1.0]],
    times=[0.0, 0.1, 0.7, 0.8, 1.5],
)
```

Playback follows a monotonic browser clock and selects the most recent
supplied frame at or before the current physical time. The frame slider
snaps the playback clock to the selected timestamp. Pause preserves the
elapsed time, Play resumes, and playback stops at the final configuration.
Pressing Play after completion restarts at the beginning. Camera operations
and Fit View do not reset playback.

If the browser is slow or a tab has been throttled, frames may be skipped
to catch up to the physical clock. This is visual playback, **not a
real-time control guarantee**. The time indicator displays elapsed
physical time and the playback-speed factor; the slider remains indexed
by frames.

Do not combine explicit `times` with `interval`. A non-default
`playback_speed` without `times` is also invalid. The original
`interval`-based mode remains unchanged and continues looping frames
uniformly by default.

**Matplotlib limitation:** physical-time playback via `times=` is **not
supported** in this version. Passing it with
`backend="matplotlib"` raises a clear `ValueError`. Matplotlib
continues to support the existing `interval`-based `FuncAnimation`
workflow; setting a fixed `interval` cannot accurately reproduce
arbitrary nonuniform physical times.

### Backend capability summary

| Animation capability | Three.js (default) | Matplotlib |
| --- | --- | --- |
| Fixed `interval` playback | Yes; loops | Yes; `FuncAnimation` |
| Explicit `times` in seconds | Yes; including nonuniform grids | Not supported (raises `ValueError`) |
| `playback_speed` with `times` | Yes; finite positive multiplier | Not supported |
| Pause / resume / frame seek | Yes; synchronized with physical time | Unchanged Matplotlib controls |
| Physical-time end behavior | Stops at the final configuration | Not applicable |
| Interpolation of missing samples | No | No T-06 feature |
| Hard real-time scheduling | Not guaranteed | Not guaranteed |

For an executable two-joint example (including a nonuniform
multi-waypoint trajectory, a trapezoidal trajectory and a visual
verification checklist), see
[Physical-Time Animation Playback](../examples/physical-time-playback.md).
The behavior is based on a browser monotonic clock; if a notebook
viewer does not render correctly, check browser console output and
whether Three.js dependencies loaded, rather than assuming
`times` was converted into a fixed `interval`.

## Customizing the visualization

Visual appearance is configured through:

```python id="r8ryak"
from moro.visualization import VisualizationStyle
```

For example:

```python id="4k8nzp"
style = VisualizationStyle(
    show_frames=False,
    show_grid=True,
    link_linewidth=5,
)
```

Then pass the same style object to either backend:

```python id="x69hxj"
viz.plot(
    values,
    backend="matplotlib",
    style=style,
)
```

or:

```python id="wcc6zb"
viz.plot(
    values,
    backend="threejs",
    style=style,
)
```

### Visibility options

The main visibility controls are:

```python id="b9fnfg"
VisualizationStyle(
    show_frames=True,
    show_links=True,
    show_joints=True,
    show_base=True,
    show_grid=True,
)
```

For example, to display only the robot links and joints:

```python id="rauwci"
style = VisualizationStyle(
    show_frames=False,
    show_grid=False,
)
```

### Colors

The default colors can also be changed:

```python id="hi89g3"
style = VisualizationStyle(
    link_color="#444444",
    joint_color="#d62728",
    base_color="#000000",
)
```

### Object sizes

The following properties control the size of the displayed elements:

```python id="hjhrce"
frame_scale
joint_size
base_size
link_linewidth
```

When:

```python id="lnrmko"
frame_scale=None
joint_size=None
base_size=None
```

their values are derived automatically from the dimensions of the robot scene.

This makes the default visualization scale appropriately for robots of different sizes.

## Animating robot configurations

A sequence of robot configurations can be animated with:

```python id="tdxcft"
viz.animate(...)
```

The input is a sequence of substitution dictionaries.

For example:

```python id="8ychbd"
configurations = [
    {
        l1: 1.0,
        l2: 0.8,
        q1: 0.0,
        q2: 0.0,
    },
    {
        l1: 1.0,
        l2: 0.8,
        q1: 0.2,
        q2: 0.1,
    },
    {
        l1: 1.0,
        l2: 0.8,
        q1: 0.4,
        q2: 0.2,
    },
]
```

Then:

```python id="ecbpmg"
animation = viz.animate(
    configurations,
)
```

Three.js is the default backend. The result is an interactive
`IPython.display.HTML` viewer. Select Matplotlib explicitly when you
need a `FuncAnimation` object.

An empty configuration sequence is not valid.

## Matplotlib animations

Use:

```python id="n96hrz"
animation = viz.animate(
    configurations,
    backend="matplotlib",
)
```

The result is a:

```text id="ykl6mi"
matplotlib.animation.FuncAnimation
```

Keep a reference to the returned animation object until it has been displayed or saved.

For example:

```python id="cmy09x"
animation = viz.animate(
    configurations,
    backend="matplotlib",
)
```

rather than calling `viz.animate(...)` without assigning the result.

### Frame interval

The default interval is:

```python id="1w6gep"
interval=100
```

milliseconds.

It can be changed with:

```python id="jqdnqt"
animation = viz.animate(
    configurations,
    backend="matplotlib",
    interval=50,
)
```

The spatial limits of the animation are computed from the complete sequence, so the camera scale remains fixed while the robot moves.

## Three.js animations

Use:

```python id="4zug0q"
viz.animate(
    configurations,
    backend="threejs",
)
```

The result is an interactive HTML animation.

The viewer includes controls for:

* Play and Pause;
* direct frame selection;
* a frame slider;
* Front, Right, Top, and Isometric views;
* orthographic and perspective cameras;
* Fit View (framing only; leaves playback and orientation unchanged);
* free orbit navigation.

The animation uses one global spatial scale for the complete sequence, so the apparent sizes of links, joints, and coordinate frames remain consistent throughout the motion.

## Showing the end-effector trajectory

Animations can optionally display the Cartesian path followed by the end-effector.

Create a style with:

```python id="nywoif"
style = VisualizationStyle(
    show_trajectory=True,
)
```

and use it in the animation:

```python id="vpby6c"
viz.animate(
    configurations,
    backend="threejs",
    style=style,
)
```

The same option is supported by the Matplotlib backend.

The displayed trajectory corresponds to the positions of the final robot frame throughout the animation sequence.

### Trajectory display modes

Two display modes are available:

```text id="4gt0cj"
"full"
"trace"
```

The default is:

```python id="z5zhzt"
trajectory_mode="full"
```

With `"full"`, the complete end-effector trajectory is displayed.

With:

```python id="511u47"
style = VisualizationStyle(
    show_trajectory=True,
    trajectory_mode="trace",
)
```

the displayed path grows as the animation advances.

Trajectory appearance can also be customized with:

```python id="76assf"
trajectory_color
trajectory_linewidth
```

For example:

```python id="j2lq29"
style = VisualizationStyle(
    show_trajectory=True,
    trajectory_color="#222222",
    trajectory_linewidth=3,
)
```

## Animating an inverse-kinematics trajectory

`RobotVisualizer.animate()` expects substitution dictionaries, while `solve_position_trajectory()` returns joint vectors through:

```python id="gspk4f"
trajectory.qs
```

The two APIs can be connected easily.

Suppose an IK trajectory has already been computed:

```python id="i0xi4t"
trajectory = solve_position_trajectory(
    robot,
    targets,
    q0=[0.1, 0.1],
    parameters={
        l1: 1.0,
        l2: 0.8,
    },
)
```

First verify that the trajectory converged:

```python id="qx0k56"
if trajectory.converged:
    print("Trajectory solved.")
```

Then convert each joint vector into a substitution dictionary:

```python id="wmug3e"
configurations = [
    dict(zip(robot.qs, q))
    for q in trajectory.qs
]
```

For a robot with symbolic geometric parameters, include them in each configuration:

```python id="4p45in"
parameters = {
    l1: 1.0,
    l2: 0.8,
}

configurations = [
    {
        **parameters,
        **dict(zip(robot.qs, q)),
    }
    for q in trajectory.qs
]
```

Now animate the result:

```python id="uqi0px"
style = VisualizationStyle(
    show_trajectory=True,
    trajectory_mode="trace",
)

viz.animate(
    configurations,
    backend="threejs",
    style=style,
)
```

This provides a convenient workflow from Cartesian position targets to a visual representation of the corresponding robot motion.

## A worked example

Consider a planar 2R robot:

```python id="t5llra"
import numpy as np

from moro import Robot
from moro.abc import q1, q2, l1, l2
from moro.visualization import (
    RobotVisualizer,
    VisualizationStyle,
)

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)

viz = RobotVisualizer(robot)
```

Define the geometric parameters and one configuration:

```python id="yc8zvw"
values = {
    l1: 1.0,
    l2: 0.8,
    q1: np.pi / 4,
    q2: -np.pi / 6,
}
```

A static Matplotlib plot can be created with:

```python id="bdzyp7"
fig, ax = viz.plot(
    values,
    backend="matplotlib",
    view_init=(35, 35),
)
```

The same configuration can be inspected interactively with Three.js:

```python id="c5eo8m"
viz.plot(
    values,
    backend="threejs",
)
```

Now create a sequence of configurations:

```python id="1fgtpf"
configurations = []

for theta in np.linspace(0, np.pi / 2, 40):
    configurations.append({
        l1: 1.0,
        l2: 0.8,
        q1: theta,
        q2: -theta / 2,
    })
```

Create a visualization style that shows the end-effector path:

```python id="gh6c0v"
style = VisualizationStyle(
    show_trajectory=True,
    trajectory_mode="trace",
)
```

Then animate the robot:

```python id="02b6dy"
viz.animate(
    configurations,
    backend="threejs",
    style=style,
)
```

The resulting viewer allows the animation to be played, paused, inspected frame by frame, and viewed using different camera projections.

The same data can be rendered with Matplotlib:

```python id="rlhh4g"
animation = viz.animate(
    configurations,
    backend="matplotlib",
    interval=75,
    style=style,
)
```

## Return types and migration from previous versions

The default backend changed in Moro 0.6.0. A bare `viz.plot(values)`
returns HTML rather than `(fig, ax)`, and bare
`viz.animate(configurations)` returns HTML rather than
`FuncAnimation`. Older notebooks that manipulate Matplotlib axes or
animation objects should request `backend="matplotlib"` explicitly.
Arguments `ax`, `figsize`, and `view_init` remain specific to
Matplotlib; `width` and `height` remain specific to Three.js. These
are not converted automatically.

## Notes and limitations

The visualization module is intended primarily for inspecting kinematic robot models and computed joint trajectories.

Keep the following points in mind:

* visualization operates on the `Robot` kinematic model;
* all symbolic quantities required by the robot transformations must be assigned numerical values;
* `RobotVisualizer.plot()` renders one configuration;
* `RobotVisualizer.animate()` renders a sequence of configurations;
* Three.js is the default backend for `plot()` and `animate()` and
  returns interactive HTML, especially convenient in notebook environments;
* Matplotlib returns normal figures or `FuncAnimation` when explicitly selected;
* Three.js preset views and camera types are controlled from the viewer interface rather than through `RobotVisualizer.plot()` arguments;
* visualization styles can be shared between both backends;
* automatic scene scaling is used when explicit object sizes are not provided;
* animation scaling remains fixed across the complete sequence;
* the optional trajectory display corresponds to the Cartesian path of the final robot frame.

The current visualization module represents robots using kinematic links, joints, and coordinate frames. It is not intended to provide detailed CAD geometry or physically realistic rendering.

The visualization system does not currently provide:

* collision detection;
* contact visualization;
* physics simulation;
* URDF or mesh-based robot geometry;
* automatic trajectory generation;
* interactive joint sliders;
* dynamic-force or torque visualization.

These capabilities may require additional modeling or visualization infrastructure beyond the current scope.

## See also

* **Robot Modeling** — define the serial robot to be visualized.
* **Forward Kinematics** — compute the frame transformations used by the visualization system.
* **Inverse Kinematics** — generate joint configurations for Cartesian targets and trajectories.
* **Jacobians** — inspect differential kinematic quantities associated with the robot motion.
* **API Reference → Visualization** — complete reference for `RobotVisualizer`, `VisualizationStyle`, backend classes, and scene-data types.
