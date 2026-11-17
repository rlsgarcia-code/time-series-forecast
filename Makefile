.PHONY: install lab test validate validate-01 validate-02 validate-03 validate-04 validate-05 validate-06 validate-07 validate-08 validate-09 validate-10 validate-11 validate-12 validate-13 validate-14 check rebuild-notebook rebuild-notebooks

install:
	uv sync

lab:
	uv run jupyter lab

test:
	uv run pytest

validate: validate-01 validate-02 validate-03 validate-04 validate-05 validate-06 validate-07 validate-08 validate-09 validate-10 validate-11 validate-12 validate-13 validate-14

validate-01:
	uv run python scripts/validate_notebook.py notebooks/01_processos_estocasticos.ipynb

validate-02:
	uv run python scripts/validate_notebook.py notebooks/02_estimacao_diagnostico_ar1.ipynb

validate-03:
	uv run python scripts/validate_notebook.py notebooks/03_arp_maq_arma.ipynb

validate-04:
	uv run python scripts/validate_notebook.py notebooks/04_nao_estacionariedade_arima.ipynb

validate-05:
	uv run python scripts/validate_notebook.py notebooks/05_browniano_gbm_ito.ipynb

validate-06:
	uv run python scripts/validate_notebook.py notebooks/06_arch_garch.ipynb

validate-07:
	uv run python scripts/validate_notebook.py notebooks/07_backtesting_selecao.ipynb

validate-08:
	uv run python scripts/validate_notebook.py notebooks/08_sistema_integrado.ipynb

validate-09:
	uv run python scripts/validate_notebook.py notebooks/09_estado_espaco_kalman.ipynb

validate-10:
	uv run python scripts/validate_notebook.py notebooks/10_markov_switching.ipynb

validate-11:
	uv run python scripts/validate_notebook.py notebooks/11_change_point.ipynb

validate-12:
	uv run python scripts/validate_notebook.py notebooks/12_cusum_ewma.ipynb

validate-13:
	uv run python scripts/validate_notebook.py notebooks/13_sistema_final.ipynb

validate-14:
	uv run python scripts/validate_notebook.py notebooks/14_ciclo_completo_usuario.ipynb

check: test validate

rebuild-notebook: rebuild-notebooks

rebuild-notebooks:
	uv run python scripts/build_notebook_01.py
	uv run python scripts/build_notebook_02.py
	uv run python scripts/build_notebook_03.py
	uv run python scripts/build_notebook_04.py
	uv run python scripts/build_notebook_05.py
	uv run python scripts/build_notebook_06.py
	uv run python scripts/build_notebook_07.py
	uv run python scripts/build_notebook_08.py
	uv run python scripts/build_notebook_09.py
	uv run python scripts/build_notebook_10.py
	uv run python scripts/build_notebook_11.py
	uv run python scripts/build_notebook_12.py
	uv run python scripts/build_notebook_13.py
	uv run python scripts/build_notebook_14.py

