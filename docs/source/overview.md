# Overview

## What is moro?

`moro` is a Python library for modeling, analyzing, and visualizing serial robotic manipulators. It is designed primarily for educational use, with an emphasis on keeping the connection between the mathematical formulation of robot kinematics and dynamics and their computational implementation as clear as possible.

The library provides tools for working with symbolic and numerical models of manipulators composed of revolute and prismatic joints. Its main capabilities include transformation matrices, forward kinematics, Jacobian computation, numerical inverse kinematics, inverse dynamics, and robot visualization.

`moro` relies extensively on [SymPy](https://www.sympy.org/) for symbolic computation, making it possible to inspect, manipulate, simplify, and evaluate the mathematical expressions generated during the analysis of a robot.

Rather than hiding the underlying mathematics behind a highly abstract interface, `moro` aims to expose the quantities commonly encountered in robotics courses and textbooks in a form that can be explored directly from Python.

## Main capabilities

`moro` currently provides tools for:

* modeling serial robotic manipulators with revolute and prismatic joints;
* working with rotation matrices and homogeneous transformations;
* defining manipulators using Denavit-Hartenberg parameters;
* computing forward kinematics;
* obtaining intermediate transformations between robot frames;
* computing geometric Jacobians;
* solving numerical inverse kinematics problems;
* solving sequences of inverse kinematics problems along position trajectories;
* deriving symbolic robot dynamics using Euler-Lagrange equations and the standard matrix form;
* evaluating symbolic models for specific joint configurations;
* plotting robotic manipulators;
* animating robot motion using Matplotlib and Three.js-based visualization backends.

Most kinematic and dynamic quantities can be represented symbolically, allowing the user to inspect the equations generated for a particular manipulator before substituting numerical values.

## Design goals

`moro` is developed with a few guiding principles in mind.

### Educational clarity

The library is intended to complement the study of robot kinematics and dynamics. Whenever possible, its API follows the terminology and mathematical objects commonly used in robotics, such as homogeneous transformation matrices, Jacobians, joint variables, centers of mass, and equations of motion.

The goal is not only to obtain a numerical result, but also to make it possible to explore how that result is constructed.

### Symbolic-first modeling

Many operations in `moro` are based on symbolic expressions. This makes it possible to define a manipulator once and then derive expressions that depend explicitly on its joint variables and physical parameters.

For example, a forward kinematics result can be inspected symbolically, simplified, differentiated, or evaluated for a particular configuration.

### Simple workflows

Common manipulator analysis tasks should require relatively little setup. A robot can be described directly from its Denavit-Hartenberg parameters and then used to compute kinematic or dynamic quantities without requiring a separate simulation environment.

### Connection between theory and computation

The documentation is organized so that practical usage and mathematical background remain connected without being mixed unnecessarily.

The **User Guide** focuses on how to perform common tasks with `moro`, while the **Theory** section develops the mathematical foundations behind those operations.

## A minimal example

A serial manipulator can be created by specifying one Denavit-Hartenberg tuple for each joint.

For example, consider a simple planar two-link manipulator with two revolute joints:

```python
from sympy import symbols
from moro import Robot

q1, q2 = symbols("q1 q2")

robot = Robot(
    (1, 0, 0, q1, "r"),
    (1, 0, 0, q2, "r"),
)
```

Once the robot has been created, its kinematic quantities can be obtained directly from the model. For instance, the homogeneous transformation from the base frame to the end-effector can be computed from the robot definition.

Because the joint variables are symbolic, the resulting transformation remains an expression in terms of `q1` and `q2` until numerical values are substituted.

More complete examples, including visualization, Jacobians, inverse kinematics, and dynamics, are introduced in the [Quick Start](quick-start.md) and throughout the **User Guide**.

## Who is moro for?

`moro` is mainly intended for:

* students learning the fundamentals of robot kinematics and dynamics;
* instructors preparing computational examples for robotics courses;
* researchers and engineers who need compact symbolic models of serial manipulators;
* Python users who want to experiment with robot models without requiring a complete robotics simulation framework.

The library can be particularly useful when the equations themselves are important, rather than only their numerical evaluation.

## Current scope

The current scope of `moro` is centered on the analysis of serial robotic manipulators.

At present, the library supports:

* serial kinematic chains;
* revolute and prismatic joints;
* symbolic and numerical evaluation of robot models;
* forward kinematics;
* geometric Jacobians;
* numerical inverse kinematics for position problems;
* inverse kinematics along position trajectories;
* inverse dynamics;
* static robot visualization;
* robot animation.

Some capabilities that are common in larger robotics frameworks are outside the current scope of the library. These include:

* forward dynamics integration;
* collision detection;
* motion and path planning;
* robot control;
* URDF-based robot description;
* physics-based simulation of robot-environment interaction.

These limitations are intentional at the current stage of development. The focus remains on providing a clear and compact environment for learning and exploring the mathematical modeling of serial manipulators.

## Where to go next

If this is your first time using `moro`, a good starting point is:

1. [Installation](installation.md) — install the library and verify your environment.
2. [Quick Start](quick-start.md) — build and analyze your first robot model.
3. **User Guide** — explore individual features in more detail.

For complete worked problems, see the **Examples** section.

For detailed descriptions of classes, functions, parameters, and return values, see the [API Reference](api/index.rst).

For the mathematical background behind the implemented methods, see the **Theory** section.
