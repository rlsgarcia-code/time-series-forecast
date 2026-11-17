# Plano didático e técnico — Notebook 08

## Objetivo

Integrar o laboratório em uma proposta executável de sistema: contrato de dados, registry de
modelos, backtester, avaliador, seleção, artefato, monitoramento prequential e risco futuro.

## Fluxo

`dados → validação → candidatos → rolling-origin → scoring → seleção → fit final → artefato →
monitoramento → risco/alertas`.

Uma mudança de média simulada após o cutoff permite distinguir anomalia pontual, evidência
persistente de drift e probabilidade de evento futuro.

## Critérios

Cutoff e configuração preservados; previsões antigas imutáveis; artefato serializado e restaurado;
modelos intercambiáveis; monitor sem leakage; risco separado de detecção; exercícios/testes;
execução integral em kernel limpo.

