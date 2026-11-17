import json
from pathlib import Path


NOTEBOOK = Path("notebooks/02_estimacao_diagnostico_ar1.ipynb")


def test_notebook_02_has_expected_sections_and_explicit_functions() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join(
        cell.get("source", "") if isinstance(cell.get("source", ""), str)
        else "".join(cell["source"])
        for cell in payload["cells"]
    )
    for required in (
        "Estimação e diagnóstico",
        "Mínimos quadrados ordinários",
        "Incerteza dos parâmetros",
        "Teste de Ljung–Box",
        "Avaliação fora da amostra",
        "Forecast multi-step",
        "Forecast rolling",
        "Exercícios",
        "Gabarito",
        "def fit_ar1_ols",
        "def rolling_one_step_ar1",
    ):
        assert required in text
    assert not any(
        output.get("output_type") == "error"
        for cell in payload["cells"]
        for output in cell.get("outputs", [])
    )

