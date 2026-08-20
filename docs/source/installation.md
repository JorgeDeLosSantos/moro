# Installation

`moro` can be installed directly from PyPI for normal use or from the GitHub repository if you want to try the current development version.

## Requirements

`moro` requires Python 3.9 or newer.

The main runtime dependencies are:

* [SymPy](https://www.sympy.org/) for symbolic computation;
* [Matplotlib](https://matplotlib.org/) for plotting and visualization.

These dependencies are installed automatically when `moro` is installed with `pip`.

Using a virtual environment is recommended so that the dependencies required by `moro` remain isolated from other Python projects.

For example, you can create and activate a virtual environment with:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

## Installing the stable version

For most users, the recommended installation method is to install the latest stable release from PyPI:

```bash
pip install moro
```

This installs `moro` together with its required dependencies.

## Installing the development version

If you want to try the latest changes that have not yet been included in a stable release, you can install the current development version directly from the `develop` branch of the GitHub repository:

```bash
pip install git+https://github.com/JorgeDeLosSantos/moro.git@develop
```

Development versions may include new features, fixes, or API changes that are still being tested.

For contributing to the project or working with an editable local installation, see the **Development** section of the documentation.

## Verifying the installation

After installation, open a Python interpreter or notebook and import `moro`:

```python
import moro
```

You can also check the installed version:

```python
import moro

print(moro.__version__)
```

If the import succeeds and a version number is displayed, the installation is ready to use.

You can perform an additional quick check by importing the main `Robot` class:

```python
from moro import Robot
```

For a first complete example, continue with the [Quick Start](quick-start).

## Updating moro

To update an existing stable installation to the latest version available on PyPI, run:

```bash
pip install --upgrade moro
```

If you installed the development version from GitHub and want to reinstall the latest state of the `develop` branch, you can use:

```bash
pip install --upgrade --force-reinstall git+https://github.com/JorgeDeLosSantos/moro.git@develop
```

## Troubleshooting

### `moro` cannot be imported

First, verify that `moro` was installed in the same Python environment that you are currently using:

```bash
python -m pip show moro
```

Using:

```bash
python -m pip install moro
```

instead of:

```bash
pip install moro
```

can help ensure that `pip` corresponds to the Python interpreter you intend to use.

### Unsupported Python version

Check your Python version with:

```bash
python --version
```

`moro` requires Python 3.9 or newer.

### Problems after upgrading

If an existing environment contains old or incompatible package versions, creating a fresh virtual environment is often the simplest solution.

You can then reinstall `moro` with:

```bash
python -m pip install moro
```

If you encounter a problem that appears to be related to the library itself, please report it through the project's GitHub issue tracker.
