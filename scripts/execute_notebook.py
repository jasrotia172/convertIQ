"""Execute the notebook using the active interpreter, without installing a user kernel."""

import os
from pathlib import Path
import sys
import tempfile
import json
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="convertiq-kernel-") as temporary:
    kernel = Path(temporary) / "kernels" / "convertiq"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "ConvertIQ",
                "language": "python",
            }
        )
    )
    os.environ["JUPYTER_PATH"] = temporary
    os.environ["MPLCONFIGDIR"] = "/tmp/convertiq-matplotlib"
    path = root / "notebooks" / "01_convertiq_analysis.ipynb"
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook, timeout=180, kernel_name="convertiq", resources={"metadata": {"path": str(root)}}
    ).execute()
    nbformat.write(notebook, path)
    print(f"Executed {path}")
