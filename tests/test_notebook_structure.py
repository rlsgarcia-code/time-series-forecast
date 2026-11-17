import json
from pathlib import Path


NOTEBOOK = Path("notebooks/01_processos_estocasticos.ipynb")


def test_notebook_has_expected_pedagogical_sections_and_no_saved_errors() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    for required in (
        "Objetivos de aprendizagem",
        "Ruído branco",
        "Passeio aleatório",
        "AR(1)",
        "100 trajetórias",
        "ACF",
        "Choques e persistência",
        "Forecast probabilístico",
        "Exercícios",
        "Gabarito",
    ):
        assert required in text
    assert not any(
        output.get("output_type") == "error"
        for cell in payload["cells"]
        for output in cell.get("outputs", [])
    )

