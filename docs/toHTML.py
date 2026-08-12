# -*- coding: utf-8 -*-
# import os
# import subprocess

# source = "source"
# build = "."
# instr = "sphinx-build -b html "+source+" "+build
# subprocess.run(instr, shell=True)
# os.startfile(build+r"\index.html")

from pathlib import Path
import os
import subprocess

docs_dir = Path(__file__).parent
source = docs_dir / "source"
build = docs_dir / "build" / "html"

subprocess.run(
    ["sphinx-build", "-b", "html", str(source), str(build)],
    check=True,
)

os.startfile(build / "index.html")