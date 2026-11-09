"""Execute the API-002 notebook-ready inverse-dynamics example end-to-end.

Sphinx renders code fences without executing the example. Run them in order
and validate numerical results, leaving interactive browser testing manual.
"""

from pathlib import Path
import re
from unittest.mock import patch

import numpy as np
from IPython.display import HTML
from matplotlib import pyplot as plt


EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "source"
    / "examples"
    / "inverse-dynamics-trajectories.md"
)


def test_api002_rr_notebook_cells_execute():
    source = EXAMPLE.read_text(encoding="utf-8")
    cells = re.findall(r"~~~python\n(.*?)\n~~~", source, flags=re.DOTALL)
    assert len(cells) == 6, "The example should have six executable cells."

    displayed = []
    ns = {}
    with patch("IPython.display.display", side_effect=displayed.append):
        with patch("matplotlib.pyplot.show"):
            try:
                for cell in cells:
                    exec(compile(cell, str(EXAMPLE), "exec"), ns)
            finally:
                plt.close("all")

    robot = ns["robot"]
    trap = ns["trapezoidal"]
    quintic = ns["quintic"]
    tau_trap = ns["tau_trapezoidal"]
    tau_quintic = ns["tau_quintic"]

    assert trap.method == "trapezoidal"
    assert quintic.method == "quintic"
    assert trap.dof == quintic.dof == robot.dof == 2
    assert tau_trap.shape == tau_quintic.shape == (trap.samples, 2)
    assert np.all(np.isfinite(tau_trap))
    assert np.all(np.isfinite(tau_quintic))

    np.testing.assert_array_equal(trap.t, quintic.t)
    np.testing.assert_allclose(trap.q[[0, -1]], quintic.q[[0, -1]])
    np.testing.assert_allclose(tau_trap[ns["index"]], ns["reference_tau"])
    assert not np.allclose(tau_trap[0], tau_quintic[0])

    assert len(displayed) == 2
    assert all(isinstance(output, HTML) for output in displayed)
