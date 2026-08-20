# Visualization

`moro` provides visualization tools for inspecting robot configurations and animations.

The main interface is `RobotVisualizer`, which takes a `Robot` model and evaluates its symbolic kinematics at numerical configurations before passing the resulting scene to one of the available rendering backends.

Two backends are currently available:

* **Matplotlib**, for static 3D plots and Python-side animations;
* **Three.js**, for interactive browser-based visualization.

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

Because Matplotlib is the default backend, the previous call is equivalent to:

```python id="xlv49t"
viz.plot(
    values,
    backend="matplotlib",
)
```

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
"matplotlib"
"threejs"
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

A reset control restores the default view.

The preset view and camera controls belong to the Three.js interface itself; they are not arguments of `RobotVisualizer.plot()`.

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

Again, Matplotlib is the default backend.

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
* Front, Top, and Isometric views;
* orthographic and perspective cameras;
* camera reset;
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

## Notes and limitations

The visualization module is intended primarily for inspecting kinematic robot models and computed joint trajectories.

Keep the following points in mind:

* visualization operates on the `Robot` kinematic model;
* all symbolic quantities required by the robot transformations must be assigned numerical values;
* `RobotVisualizer.plot()` renders one configuration;
* `RobotVisualizer.animate()` renders a sequence of configurations;
* Matplotlib is the default backend;
* the Three.js backend returns interactive HTML and is especially convenient in notebook environments;
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
