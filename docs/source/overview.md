# Overview

## What is moro?

`moro` is a Python library for modeling, analyzing, simulating, and visualizing serial robotic manipulators. It is designed primarily for educational use, with an emphasis on keeping the connection between the mathematical formulation of robot kinematics and dynamics and their computational implementation as clear as possible.

The library combines symbolic robot models with focused numerical tools. Symbolic quantities remain available for inspection through SymPy, while numerical layers provide inverse kinematics, differential-kinematics analysis, trajectory generation, dynamics evaluation, time integration, and sampled workspace analysis.

Rather than hiding the underlying mathematics behind a highly abstract interface, `moro` aims to expose the quantities commonly encountered in robotics courses and textbooks in a form that can be explored directly from Python.

## Main capabilities

Moro 0.5.x provides tools for:

* modeling serial robotic manipulators with revolute and prismatic joints;
* working with rotation matrices, homogeneous transformations, Euler/Tait-Bryan angles, axis-angle, quaternions, and rotation vectors;
* defining manipulators using Denavit-Hartenberg parameters;
* computing symbolic forward kinematics and intermediate transformations;
* computing geometric Jacobians;
* evaluating task Jacobians and Cartesian velocities;
* solving velocity inverse kinematics;
* analyzing singular values, numerical rank, condition number, singularity, and manipulability;
* solving numerical inverse kinematics for Cartesian position and full pose;
* solving sequences of position inverse-kinematics targets;
* generating linear, cubic, and quintic trajectories in joint or Cartesian position space;
* deriving symbolic robot dynamics using Euler-Lagrange equations and the standard matrix form;
* evaluating inverse and forward dynamics numerically;
* integrating the equations of motion with numerical generalized-force inputs;
* sampling reachable Cartesian workspace over finite joint domains;
* plotting robotic manipulators and sampled workspaces;
* animating robot motion using Matplotlib and Three.js-based visualization backends.

Most kinematic and dynamic quantities can still be represented symbolically, allowing the user to inspect the equations generated for a manipulator before substituting numerical values.

## Design goals

### Educational clarity

The library is intended to complement the study of robot kinematics and dynamics. Whenever possible, its API follows the terminology and mathematical objects commonly used in robotics, such as homogeneous transformation matrices, Jacobians, joint variables, centers of mass, equations of motion, trajectories, and workspaces.

The goal is not only to obtain a numerical result, but also to make it possible to explore how that result is constructed.

### Symbolic-first modeling

`Robot` remains the central symbolic model. A manipulator can be defined once and then used to derive expressions that depend explicitly on joint variables and physical parameters.

Numerical layers evaluate or simulate those symbolic models without replacing them.

### Simple numerical workflows

Numerical APIs are intentionally focused rather than framework-like. For example, trajectory generation is independent of robot modeling, workspace sampling returns a transparent point/configuration dataset, and numerical dynamics uses standard NumPy/SciPy representations.

### Connection between theory and computation

The documentation separates practical usage from mathematical background while keeping them linked.

The **User Guide** focuses on how to perform common tasks with `moro`, while the **Theory** section develops the mathematical foundations behind those operations.

## A minimal example

A serial manipulator can be created by specifying one Denavit-Hartenberg tuple for each joint.

For example, consider a simple planar two-link manipulator with two revolute joints:

```python
from moro import Robot
from moro.abc import q1, q2

robot = Robot(
    (1, 0, 0, q1, "r"),
    (1, 0, 0, q2, "r"),
)
```

Once the robot has been created, symbolic kinematic quantities are available directly from the model:

```python
T = robot.T
J = robot.J
```

The expressions remain symbolic until numerical values are substituted or passed through one of Moro's numerical analysis layers.

More complete examples, including visualization, inverse kinematics, trajectories, dynamics, and workspace sampling, are introduced in the [Quick Start](quick-start.md), **User Guide**, and **Examples** sections.

## Who is moro for?

`moro` is mainly intended for:

* students learning robot kinematics and dynamics;
* instructors preparing computational examples for robotics courses;
* researchers and engineers who need compact symbolic/numerical models of serial manipulators;
* Python users who want to experiment with robot models without requiring a full robotics simulation framework.

The library is particularly useful when the equations and conventions themselves are important, rather than only the final numerical answer.

## Current scope

The current scope remains centered on serial robotic manipulators and educational analysis. Moro intentionally does not attempt to be a complete robot-software or physics-simulation framework.

Capabilities outside the 0.5.x scope include:

* general motion and path planning;
* collision detection and collision-aware IK;
* URDF-based robot description;
* constrained/contact dynamics;
* physics-based robot-environment interaction;
* built-in controller classes;
* general pose-trajectory interpolation such as SLERP-based orientation trajectories;
* exact analytical workspace-boundary reconstruction.

These exclusions keep the package focused on inspectable kinematics/dynamics and compact numerical analysis workflows.

## Where to go next

If this is your first time using `moro`, a good starting point is:

1. [Installation](installation.md) — install the library and verify your environment.
2. [Quick Start](quick-start.md) — build and analyze your first robot model.
3. **User Guide** — explore individual features in more detail.
4. **Examples** — follow complete workflows for IK, trajectories, dynamics, and workspace analysis.

For detailed descriptions of classes, functions, parameters, and return values, see the [API Reference](api/index.rst).

For the mathematical background behind the implemented methods, see the **Theory** section.

For migration notes and the main 0.5.0 changes, see the **Release Notes** section.
