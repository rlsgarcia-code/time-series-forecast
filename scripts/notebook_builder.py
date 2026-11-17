"""Helpers pequenos para geradores versionáveis de notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def write_notebook(path: Path, cells: list[dict], number: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"nb{number:02d}-{index:03d}"
    payload["cells"] = cells
    payload["metadata"].update(
        {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "time_series_lab": {"notebook": number, "language": "pt-BR"},
        }
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Notebook reconstruído: {path} ({len(cells)} células)")


def setup_cell(extra_imports: str = "") -> dict:
    extra_block = "\n".join(
        f"        {line}" for line in extra_imports.splitlines() if line.strip()
    )
    return code(
        f"""
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy import stats
{extra_block}

        %matplotlib inline
        COLORS = {{
            "blue": "#2563EB", "orange": "#EA580C", "green": "#059669",
            "purple": "#7C3AED", "gray": "#64748B", "red": "#DC2626",
        }}
        plt.rcParams.update({{
            "figure.figsize": (10, 4.8), "axes.spines.top": False,
            "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.20,
            "axes.titleweight": "bold", "font.size": 11,
        }})
        print("Ambiente pronto — NumPy", np.__version__)
        """
    )

