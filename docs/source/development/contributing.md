# Contributing

Contributions to `moro` are welcome.

`moro` is primarily an educational and research-oriented robotics library, so contributions should aim to preserve a clear API, readable symbolic formulations, and behavior that is useful for teaching and analysis.

This page summarizes the recommended development workflow for contributing code, documentation, tests, or examples.

## Development setup

Clone the repository:

```bash
git clone https://github.com/JorgeDeLosSantos/moro.git
cd moro
```

The active development branch is:

```text
develop
```

Switch to it before starting new work:

```bash
git checkout develop
git pull
```

It is recommended to create a dedicated branch for each change:

```bash
git checkout -b feature/my-change
```

or, for documentation-only work:

```bash
git checkout -b docs/my-change
```

## Python environment

`moro` requires Python 3.9 or newer.

Creating a virtual environment is recommended.

For example:

```bash
python -m venv .venv
```

Activate it using the appropriate command for your operating system.

On Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Then install the project in editable mode:

```bash
pip install -e .
```

Editable installation allows changes made to the local source code to become immediately available when importing `moro`.

The main runtime dependencies are installed automatically with the package.

## Installing development tools

The test suite uses `pytest`:

```bash
pip install pytest
```

To build the documentation locally, install the documentation dependencies:

```bash
pip install sphinx myst-parser sphinx-rtd-theme numpydoc
```

## Repository structure

The main project directories are:

```text
moro/
├── moro/
├── tests/
├── examples/
└── docs/
```

The `moro/` package contains the library source code.

The `tests/` directory contains the automated test suite.

The `examples/` directory contains example notebooks and related material.

The `docs/` directory contains the Sphinx documentation.

The current documentation sources live in:

```text
docs/source/
```

and generated HTML files are written to:

```text
docs/build/html/
```

Generated documentation should not be committed to the repository.

## Running the tests

Run the complete test suite from the repository root:

```bash
pytest
```

The current tests cover the main library areas, including:

```text
tests/
├── test_core.py
├── test_transformations.py
├── test_inverse_kinematics.py
└── test_visualization.py
```

A specific test module can be executed independently.

For example:

```bash
pytest tests/test_transformations.py
```

or:

```bash
pytest tests/test_inverse_kinematics.py
```

When adding a new feature or fixing a bug, add or update tests whenever the behavior can be verified automatically.

Before submitting a contribution, the complete test suite should pass.

## Writing tests

Tests should focus on observable behavior rather than implementation details whenever possible.

For symbolic robotics functionality, useful checks may include:

- exact symbolic equality;
- simplified symbolic equality;
- matrix dimensions;
- numerical evaluation at known configurations;
- validation of input errors;
- convergence behavior for numerical algorithms;
- preservation of documented invariants.

Bug fixes should ideally include a regression test that fails before the fix and passes afterward.

When testing numerical algorithms, avoid relying unnecessarily on a single exact iteration count or floating-point representation unless that behavior is explicitly part of the API.

## Building the documentation

The documentation uses Sphinx with both reStructuredText and MyST Markdown sources.

Build the HTML documentation from the repository root with:

```bash
sphinx-build -b html docs/source docs/build/html
```

The generated site can then be opened from:

```text
docs/build/html/index.html
```

For a stricter validation before merging documentation changes, use:

```bash
sphinx-build -W -b html docs/source docs/build/html
```

The `-W` option treats warnings as errors and helps detect problems such as:

- broken references;
- missing documents;
- invalid directives;
- autodoc failures;
- malformed MyST content.

## Documentation structure

The documentation is organized into several layers:

```text
Getting Started
User Guide
Examples
API Reference
Theory
Development
```

These sections serve different purposes.

**Getting Started** should help a new user install `moro` and perform a first analysis quickly.

**User Guide** pages should explain how to use specific parts of the library in practice.

**Examples** should present complete robotics workflows rather than isolated API calls.

**API Reference** is generated primarily from public docstrings using Sphinx autodoc.

**Theory** explains the mathematical concepts and conventions behind the implementation.

**Development** contains information intended for contributors.

When adding documentation, prefer extending the appropriate existing layer instead of duplicating the same explanation in several places.

## Documentation style

The documentation is written in English.

Narrative documentation is primarily written in Markdown using MyST:

```text
*.md
```

Sphinx structural pages and API Reference pages may use reStructuredText:

```text
*.rst
```

Examples should favor readable symbolic models and small, reproducible numerical cases.

Code shown in the documentation should reflect the current public API.

Whenever possible, documentation examples should be simple enough to execute independently.

## Naming conventions

Contributors should follow the naming and notation conventions adopted by the project.

See:

```text
Development → Naming Conventions
```

These conventions are especially important because robotics notation can become ambiguous when frame indices, coordinate systems, transformations, positions, and joint variables are mixed.

New APIs should remain consistent with the existing conventions unless there is a strong reason to introduce a new one.

Any intentional deviation should be clearly documented.

## Public API

Changes to the public API should be made carefully.

Public functionality currently includes the main robot model, transformation utilities, inverse-kinematics tools, and visualization interfaces.

When modifying public behavior:

1. preserve backward compatibility when practical;
2. update the corresponding docstrings;
3. update User Guide or Examples when behavior changes materially;
4. add or update tests;
5. record user-visible changes in the changelog when appropriate.

Internal implementation details should not become public API accidentally.

## Docstrings

Public classes, methods, and functions should have clear docstrings.

Docstrings should explain at least:

- purpose;
- parameters;
- return values;
- important assumptions;
- possible exceptions;
- relevant conventions.

For mathematical functions, document frame conventions, coordinate representations, sequence conventions, or numerical assumptions whenever they affect interpretation.

The API Reference is generated from these docstrings, so incomplete docstrings directly affect the published documentation.

## Symbolic computations

`moro` relies heavily on SymPy.

When contributing symbolic algorithms, consider both mathematical correctness and expression complexity.

Avoid unnecessary expansion or simplification when it introduces substantial computational cost without improving the result.

Prefer formulations that preserve readable expressions and reuse already-computed kinematic quantities when possible.

Changes that alter symbolic output should be tested carefully because mathematically equivalent expressions may have different computational costs.

## Numerical algorithms

Numerical algorithms should expose convergence information when appropriate.

For iterative methods, consider:

- tolerance;
- iteration limits;
- initialization;
- reproducibility;
- stagnation;
- numerical singularities;
- joint limits;
- failure diagnostics.

A numerical method should not silently report success when its convergence condition has not been satisfied.

When randomness is involved, provide a reproducible mechanism whenever practical.

## Visualization

Visualization changes should preserve the separation between:

```text
robot model
        ↓
numerical scene evaluation
        ↓
rendering backend
```

Backend-specific behavior should remain isolated whenever practical.

Changes to Matplotlib or Three.js rendering should be checked with representative planar and spatial manipulators.

Animations should also be checked for configurations whose geometric scale changes substantially across the sequence.

## Adding dependencies

`moro` intentionally has a small dependency set.

Before introducing a new mandatory dependency, consider whether the same functionality can reasonably be implemented using the existing stack.

A new dependency should provide a clear benefit to the library and should not significantly complicate installation for educational users.

Optional functionality should remain optional whenever practical.

## Changelog

User-visible changes should be recorded in:

```text
CHANGELOG.md
```

Examples include:

- new public features;
- changes in behavior;
- bug fixes;
- deprecated functionality;
- compatibility changes;
- important documentation additions.

Small internal refactors that do not affect users generally do not require a changelog entry.

## Submitting changes

Before submitting a contribution:

1. update your branch with the latest `develop`;
2. run the complete test suite;
3. build the documentation if the change affects public behavior or docs;
4. review the diff for unrelated modifications;
5. ensure new public behavior is documented;
6. update `CHANGELOG.md` when appropriate.

A typical final verification may include:

```bash
pytest
```

and:

```bash
sphinx-build -W -b html docs/source docs/build/html
```

Commits should be focused and use concise messages describing the change.

Examples:

```text
Add stagnation detection to CCD solver
Fix frame scaling in Three.js animation
Document position IK trajectory solver
```

## Pull requests

Pull requests should target the `develop` branch unless a different target has been explicitly agreed upon.

A pull request should explain:

- what was changed;
- why the change is useful or necessary;
- any important implementation decisions;
- how the change was tested;
- whether documentation or public behavior changed.

Large contributions are easier to review when they are divided into logically coherent changes.

When introducing a substantial feature, discussing the design before implementing a large patch is encouraged.

## Reporting issues

Bug reports should include enough information to reproduce the problem.

Useful information includes:

- `moro` version;
- Python version;
- operating system when relevant;
- minimal robot definition;
- minimal code reproducing the issue;
- expected behavior;
- actual behavior;
- traceback or error message.

For numerical or symbolic problems, include the specific parameter values and configuration whenever possible.

## Scope of contributions

`moro` is primarily focused on serial robot manipulators and on educationally useful tools for their analysis.

Contributions that fit naturally within this scope include:

- kinematic modeling;
- transformations;
- Jacobians;
- inverse kinematics;
- dynamics;
- visualization;
- examples;
- tests;
- documentation.

Larger features should preserve the project's educational character and avoid adding complexity without a clear robotics use case.
