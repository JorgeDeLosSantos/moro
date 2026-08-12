# Moro Naming Conventions

This document defines naming conventions for public APIs in Moro, especially for quantities associated with serial robot kinematics and dynamics.

The goal is to keep the API:

* mathematically meaningful;
* predictable;
* suitable for teaching;
* consistent as the library grows;
* close to common robotics notation without sacrificing readability.

These conventions should guide the design of new public methods and properties.

> Some legacy API names may not fully follow these conventions. They can be retained temporarily for backward compatibility and migrated gradually when justified.

---

## 1. General principles

Moro uses a hybrid naming style.

Short mathematical names are preferred when the corresponding symbol is standard and widely recognizable in robotics, for example:

```python
T
J
q(i)
w(i)
m(i)
```

More descriptive names are preferred for higher-level quantities, collections, configuration parameters, and operations, for example:

```python
joint_limits
inertia_matrix()
potential_energy()
cm_positions
```

The API should not force every quantity into either purely mathematical or purely descriptive naming. Closeness to the notation commonly used in robotics textbooks is considered valuable, particularly because Moro is intended for educational use.

---

## 2. Default frame convention

For vector and physical quantities returned by `Robot`, the base frame `{0}` is considered the default frame of expression unless the method documentation explicitly states otherwise.

Therefore, the base frame does not normally need to appear in the method name.

Examples:

```python
z(i)
r_o(i)
r_cm(i)
w(i)
```

These quantities are understood to be expressed with respect to the base frame `{0}`.

Future methods should follow the same convention when the base frame is the natural and unambiguous representation.

For example, a relative angular velocity expressed in the base frame should preferably be named:

```python
w_rel(i)
```

rather than:

```python
w_rel0(i)
```

provided that no alternative frame representation needs to be distinguished.

---

## 3. Transformations and rotations between frames

Transformations and rotation matrices are treated differently from ordinary vectors.

When a matrix represents an explicit relationship between coordinate frames, the relevant frame indices should remain part of the name.

Examples:

```python
T_ij(i, j)
T_i0(i)
R_i0(i)
```

Here, the frame indices are not merely indicating the default frame of expression. They describe the relationship represented by the matrix itself.

Therefore, names such as:

```python
T_i0
R_i0
```

should not be simplified merely because `{0}` is the base frame.

---

## 4. Quantities that have meaningful representations in different frames

When the same physical quantity has multiple natural representations in different frames, the frame should be made explicit in the API.

Prefer descriptive suffixes such as:

```text
_local
_base
```

rather than compact numeric suffixes when the descriptive form improves clarity.

For example, an inertia tensor at the center of mass may have two distinct meanings:

```python
I_cm_local(i)
I_cm_base(i)
```

where:

* `I_cm_local(i)` represents the inertia tensor at the center of mass expressed in the link-local orientation;
* `I_cm_base(i)` represents the same physical tensor expressed in the orientation of the base frame.

This distinction is important because `cm` identifies the reference point of the tensor but does not, by itself, identify the orientation of the coordinate axes used to express it.

Names such as:

```python
I_cm(i)
I_cm0(i)
```

may therefore be considered legacy forms if more explicit names are introduced in the future.

---

## 5. Avoid redundant symbolic indices

A symbolic index should appear in a method name only when it conveys information that is not already evident from the method arguments or semantics.

For example:

```python
J_cm_i(i)
Jv_cm_i(i)
Jw_cm_i(i)
```

contain an `_i` suffix even though the link index is already supplied through the `i` argument.

For future APIs, prefer:

```python
J_cm(i)
Jv_cm(i)
Jw_cm(i)
```

unless the additional index is needed to distinguish mathematically different quantities.

By contrast:

```python
T_ij(i, j)
```

should retain `ij`, because the presence of two frame indices is fundamental to the meaning of the transformation.

---

## 6. Scalar accessors and collections

When a quantity naturally exists both as an individual value and as a collection, concise scalar accessors and descriptive plural properties may coexist.

Examples:

```python
q(i)
qs

m(i)
masses

joint_type(i)
joint_types
```

This pattern is acceptable when the short form corresponds naturally to mathematical notation.

New APIs should avoid introducing unnecessary alternate names for the same concept.

---

## 7. Time derivatives

Time derivatives should use descriptive suffixes when this improves readability.

For example:

```python
q_dot(i)
```

is preferred over alternatives such as:

```python
qd(i)
dq(i)
```

because `q_dot` corresponds directly to the conventional notation (\dot{q}) while remaining readable in Python.

The same convention may be used for future quantities where a time derivative needs to be exposed explicitly.

---

## 8. Center-of-mass notation

Use `cm` consistently as the abbreviation for center of mass.

Examples:

```python
r_cm(i)
cm_positions
J_cm(i)
Jv_cm(i)
Jw_cm(i)
I_cm_local(i)
```

Avoid introducing alternative abbreviations such as:

```text
com
cg
mass_center
```

unless a compelling compatibility reason exists.

---

## 9. Linear and angular Jacobians

When distinguishing the linear and angular blocks of a geometric Jacobian, use:

```text
Jv
Jw
```

respectively.

Examples:

```python
Jv_cm(i)
Jw_cm(i)
```

This follows common robotics notation:

[
J =
\begin{bmatrix}
J_v \
J_\omega
\end{bmatrix}.
]

The complete geometric Jacobian may continue to use:

```python
J
J_point(...)
J_cm(...)
```

---

## 10. Public names should describe semantic distinctions

Different public names should correspond to meaningful mathematical or behavioral distinctions.

Avoid creating separate names merely because two methods are implemented differently internally.

Conversely, when two methods return physically different representations, the naming should make that distinction visible.

For example:

```python
I_cm_local(i)
I_cm_base(i)
```

are justified as separate names because the tensors are expressed in different orientations.

---

## 11. Prefer readability over excessive compactness

Compact mathematical notation is useful, but names should not become cryptic.

A name such as:

```python
w_rel(i)
```

is preferable to a longer form such as:

```python
relative_angular_velocity(i)
```

because `w` is conventional and widely understood in the context of Moro.

However, a form such as:

```python
I_cm_base(i)
```

may be preferable to:

```python
I_cm0(i)
```

when the latter creates ambiguity.

The preferred balance is:

> Use compact mathematical notation for the quantity itself and descriptive qualifiers for distinctions that may otherwise be ambiguous.

---

## 12. Current API examples

The following existing names are consistent with these conventions and should generally be retained:

```python
T
T_ij(i, j)
T_i0(i)
R_i0(i)

z(i)
r_o(i)
r_cm(i)

J
J_point(point, i)

q(i)
q_dot(i)

w(i)

m(i)
masses

joint_type(i)
joint_types
joint_limits

inertia_matrix()
coriolis_matrix()
gravity_vector()

kinetic_energy()
potential_energy()
lagrangian()
```

The following existing names may be candidates for future migration:

```python
w_rel0(i)      -> w_rel(i)

J_cm_i(i)      -> J_cm(i)
Jv_cm_i(i)     -> Jv_cm(i)
Jw_cm_i(i)     -> Jw_cm(i)

I_cm(i)        -> I_cm_local(i)
I_cm0(i)       -> I_cm_base(i)
```

These are recommendations for future API evolution and do not imply that the legacy forms must be removed immediately.

---

## 13. Backward compatibility

Public API names should not be renamed casually.

When a clearer name is introduced for an existing method, prefer a gradual migration:

1. introduce the new name;
2. retain the old name temporarily as an alias;
3. document the preferred name;
4. optionally emit a deprecation warning in a later release;
5. remove the legacy name only in a release where the compatibility impact is justified.

This is especially important for APIs likely to appear in:

* teaching material;
* notebooks;
* examples;
* research code;
* published documentation.

---

## 14. Checklist for new API names

Before introducing a new public method or property, verify:

* [ ] Is the mathematical symbol standard enough to justify a short name?
* [ ] Is the quantity expressed in the base frame by default?
* [ ] If `{0}` is only the default frame of expression, can it be omitted from the name?
* [ ] Does the quantity have meaningful representations in multiple frames?
* [ ] If so, should `_local`, `_base`, or another descriptive qualifier be used?
* [ ] Are frame indices mathematically essential to the quantity, as in `T_ij`?
* [ ] Does the name contain an index already supplied as a method argument?
* [ ] Is the abbreviation already used elsewhere in Moro?
* [ ] Does the new name remain understandable in an educational context?
* [ ] Could the name conflict with or obscure an existing public API concept?
* [ ] Does introducing the name require a backward-compatibility or deprecation plan?

---

## Summary

The central naming rule for frame-dependent quantities in Moro is:

> **The base frame ****`{0}`**** is the default frame of expression for vector and physical quantities and is normally omitted from their names. Frame indices are retained when they define an explicit relationship between coordinate frames. When multiple frame representations of the same quantity are meaningful, use explicit descriptive qualifiers such as ****`_local`**** and ****`_base`****.**

This convention should be used as the default guideline for future Moro API development.
