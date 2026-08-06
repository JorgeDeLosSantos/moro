# moro

[![PyPI version](https://img.shields.io/pypi/v/moro.svg)](https://pypi.org/project/moro/)
[![License](https://img.shields.io/github/license/JorgeDeLosSantos/moro.svg)](https://github.com/JorgeDeLosSantos/moro/blob/master/LICENSE)

A Python library for kinematic and dynamic (symbolic) modeling of robots.

## Features

* **Transformations:** Use `SO(3)` rotation matrices and `SE(3)` homogeneous transformation matrices.
* **Forward kinematics:** Easily compute forward kinematics using Denavit-Hartenberg parameters.
* **Differential kinematics:** Compute the jacobian matrix.
* **Dynamic modeling:** Derive equations of motion symbolically using the Euler–Lagrange formulation.

## Installation

Install the latest stable version from **PyPI**:

```
pip install moro
```

Or install the development version directly from the GitHub repository:

```
pip install git+https://github.com/JorgeDeLosSantos/moro.git
```

## Quick Start

Here is a quick example showing how easy it is to create a 2R planar robot and derive its dynamic model using the Euler–Lagrange formulation:

```python
import moro as mr
from moro import m1,m2,l1,l2,lc1,lc2,q1,q2,g

RR = mr.Robot((l1,0,0,q1,"r"), (l2,0,0,q2,"r"))

RR.masses = [m1,m2] 
RR.inertia_tensors = RR.generate_diagonal_inertia_tensors()
rG11 = [-(l1-lc1),0,0] # CoM of link 1 in {1}-frame
rG22 = [-(l2-lc2),0,0] # CoM of link 2 in {2}-frame
RR.cm_positions = [rG11,rG22] 
RR.gravity = [0,-g,0] # gravity acc. in {0}-frame

RR.dynamic_model_matrix_form() # M*qdd + C*qd + G = tau
```


## Documentation and examples

- Check out the full API reference at [https://jorgedelossantos.github.io/moro/](https://jorgedelossantos.github.io/moro/). 

- Examples:
    - Forward kinematics [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I2rTyGTYck_WBwpO9DM2Ho4iq4prefmC?usp=sharing)
    - Inverse kinematics [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/150GvAqYKKi_C5FcysB6lhM094bxyMZ5z?usp=sharing)
    - Computing jacobian [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XTBz3UDyVml4RF-k_5snQPrGUPgCCyJf?usp=sharing)
    - Dynamic modeling (Euler-Lagrange) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12c1HqJvYESN-blMAVkJ6bZAcMFN96PTE?usp=sharing)
    - Visualization [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1s0BtRHBr3mUHngCfE9A_nlVoytDImxcr?usp=sharing)

## Roadmap

Want to know what's coming next? Check out the [Moro Roadmap Wiki.](https://github.com/jorgedelossantos/moro/wiki/Roadmap)

## Bug Reports & Support

If you encounter any bugs, have questions, or want to request a feature, please open an issue on the [GitHub Issue Tracker](https://github.com/jorgedelossantos/moro/issues). Contributions are always welcome!