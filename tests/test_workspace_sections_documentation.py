"""Execute the notebook-style workspace section example without a browser."""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def test_workspace_section_example_code_cells(monkeypatch):
    source = (
        Path(__file__).resolve().parents[1]
        / "docs/source/examples/workspace-sections.md"
    ).read_text(encoding="utf-8")
    cells = re.findall(r"\`\`\`python\\n(.*?)\\n\`\`\`", source, re.S)
    assert len(cells) == 5
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = {"__name__": "__main__"}
    try:
        for cell in cells:
            exec(compile(cell, "workspace-sections.md", "exec"), namespace)
        assert namespace["section"].samples == 80
        assert namespace["empty"].samples == 0
    finally:
        plt.close("all")
