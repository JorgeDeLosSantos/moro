"""Smoke-test the notebook-ready API-001 visualization example.

Sphinx parses code fences but does not execute notebook examples. This test
executes the documented code blocks in order, without launching a browser.
Browser-level playback remains a separate manual acceptance check.
"""

import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
from IPython.display import HTML
from matplotlib import pyplot as plt


EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "source"
    / "examples"
    / "symbolic-visualization-parameters.md"
)


def test_api001_notebook_cells_execute_without_browser():
    text = EXAMPLE.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", text, flags=re.DOTALL)
    assert len(blocks) >= 6, "Expected a standalone RR/RP notebook example."

    rendered = []
    namespace = {}
    with patch("IPython.display.display", side_effect=rendered.append):
        with patch("matplotlib.pyplot.show"):
            try:
                for code in blocks:
                    exec(compile(code, str(EXAMPLE), "exec"), namespace)
            finally:
                plt.close("all")

    assert len(rendered) >= 5
    assert all(isinstance(view, HTML) for view in rendered)
    np.testing.assert_array_equal(
        namespace["configuration"], [0.5, -0.3]
    )
    assert namespace["q_history"].shape == (5, 2)
    np.testing.assert_array_equal(
        namespace["sample_times"], [0.0, 0.1, 0.7, 0.8, 1.5]
    )
    assert namespace["parameters"] == {
        namespace["l1"]: 1.0,
        namespace["l2"]: 0.8,
    }
