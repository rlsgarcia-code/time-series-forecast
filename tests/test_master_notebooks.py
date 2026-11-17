import json
from pathlib import Path

import pytest


CASES = [
    ("03_arp_maq_arma.ipynb", ["AR(p)", "MA(q)", "ACF", "PACF", "def simulate_arma", "Exercícios"]),
    ("04_nao_estacionariedade_arima.ipynb", ["Não estacionariedade", "ADF", "Diferenciação", "ARIMA", "def difference", "Exercícios"]),
    ("05_browniano_gbm_ito.ipynb", ["Movimento Browniano", "Itô", "GBM", "def brownian_paths", "quadratic_variation", "Exercícios"]),
    ("06_arch_garch.ipynb", ["ARCH/GARCH", "GARCH(1,1)", "Ljung", "def simulate_garch11", "forecast_garch_variance", "Exercícios"]),
    ("07_backtesting_selecao.ipynb", ["Backtesting", "rolling-origin", "CRPS", "def run_backtest", "PIT", "Exercícios"]),
    ("08_sistema_integrado.ipynb", ["Sistema integrado", "SeriesContract", "backtest", "ForecastArtifact", "monitor_stream", "Exercícios"]),
    ("09_estado_espaco_kalman.ipynb", ["estado-espaço", "filtro de Kalman", "def kalman_filter_local_level", "inovação", "suavização", "Exercícios"]),
    ("10_markov_switching.ipynb", ["Markov switching", "matriz de transição", "def hamilton_filter", "smoother de Kim", "mistura", "Exercícios"]),
    ("11_change_point.ipynb", ["change-point", "CUSUM", "def best_single_change", "programação dinâmica", "penalidade", "Exercícios"]),
    ("12_cusum_ewma.ipynb", ["Detecção sequencial", "CUSUM", "EWMA", "def two_sided_cusum", "run length", "Exercícios"]),
    ("13_sistema_final.ipynb", ["Sistema integrado", "ForecastEnvelope", "SequentialMonitor", "backtest_registry", "risco de trajetória", "Exercícios"]),
    ("14_ciclo_completo_usuario.ipynb", ["Ciclo completo", "EDITE SOMENTE ESTA CÉLULA", "MISSING_VALUE_POLICY", "rolling-origin", "SelectionPolicy", "diagnose", "process_observation", "Exercícios"]),
]


@pytest.mark.parametrize(("filename", "required"), CASES)
def test_planned_notebook_is_complete_and_explicit(filename: str, required: list[str]) -> None:
    path = Path("notebooks") / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join(
        cell.get("source", "") if isinstance(cell.get("source", ""), str)
        else "".join(cell["source"])
        for cell in payload["cells"]
    )
    assert len(payload["cells"]) >= 20
    assert sum(cell["cell_type"] == "code" for cell in payload["cells"]) >= 8
    for phrase in required:
        assert phrase in text
    assert "Gabarito" in text or "testes rápidos" in text
    assert not any(
        output.get("output_type") == "error"
        for cell in payload["cells"]
        for output in cell.get("outputs", [])
    )


def test_each_planned_notebook_has_a_plan_document() -> None:
    for number in range(1, 15):
        assert Path(f"docs/plano_notebook_{number:02d}.md").exists()

