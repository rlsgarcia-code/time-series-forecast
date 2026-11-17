"""Executa um notebook do início ao fim em um diretório temporário."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--save-executed", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = (root / args.notebook).resolve() if not args.notebook.is_absolute() else args.notebook
    notebook = nbformat.read(source, as_version=4)

    with tempfile.TemporaryDirectory(prefix="time-series-lab-") as tmp:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tmp) / "matplotlib"))
        client = NotebookClient(
            notebook,
            timeout=240,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
            record_timing=True,
        )
        client.execute()
        if args.save_executed:
            destination = args.save_executed.resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(notebook, destination)

    code_cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"OK: {source.name} executado por completo ({code_cells} células de código).")


if __name__ == "__main__":
    main()

