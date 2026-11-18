# Time Series Lab — Visão do Sistema

**Status:** Motor marginal univariado e composição probabilística 01–16 concluídos
**Escopo atual:** Modelos marginais univariados e grafos derivados hierárquicos

## 1. Objetivo

Desenvolver um sistema extensível de análise temporal capaz de caracterizar séries, selecionar e ajustar modelos, produzir previsões probabilísticas, simular trajetórias futuras e identificar riscos de desvio do comportamento esperado.

O sistema deverá avaliar continuamente novas observações, reconhecer possíveis mudanças estruturais e fornecer informações para decisões preventivas.

## 2. Capacidades funcionais

### Caracterização
- Identificar tendência, sazonalidade, memória e estacionariedade.
- Caracterizar distribuições, resíduos e volatilidade.
- Produzir diagnósticos que orientem a seleção dos modelos.

### Forecasting probabilístico
- Ajustar diferentes famílias de modelos temporais.
- Produzir previsões pontuais e distribuições preditivas.
- Avaliar erro, cobertura, calibração e estabilidade.

### Simulação
- Gerar trajetórias futuras por Monte Carlo.
- Simular choques, mudanças de parâmetros e diferentes regimes.
- Estimar distribuições de resultados e eventos futuros.

### Monitoramento sequencial
- Avaliar cada observação frente à distribuição preditiva anterior.
- Identificar anomalias, desvios persistentes e possíveis mudanças de regime.
- Registrar evidências, intensidade e momento dos alertas.

### Antecipação e risco
- Estimar a probabilidade de ultrapassar limites críticos.
- Calcular a distribuição do tempo até eventos adversos.
- Comparar políticas de alerta e seus custos de falso positivo e atraso.
- Fornecer subsídios para intervenções preventivas.

## 3. Famílias e técnicas

O sistema deverá permitir a experimentação e integração de:

- Modelos AR, MA, ARMA e ARIMA.
- Modelos dinâmicos de estado-espaço e filtragem de Kalman.
- Modelos de volatilidade condicional GARCH.
- Modelos Markov switching.
- Detectores de mudanças estruturais, incluindo change point detection.
- Métodos sequenciais de detecção de drift, incluindo CUSUM e EWMA.

As famílias de modelos e os algoritmos de detecção deverão ser componentes distintos, combináveis quando estatisticamente apropriado.

## 4. Arquitetura conceitual

**Entrada da série → Caracterização → Modelagem → Forecast → Simulação → Avaliação de risco → Monitoramento → Alertas**

As previsões anteriores à chegada de novas observações deverão ser preservadas para avaliação prequential, evitando vazamento de informação.

O sistema deverá distinguir entre:

1. Anomalias em observações individuais.
2. Evidências de mudança de comportamento já iniciada.
3. Probabilidade de ocorrência de eventos adversos futuros.

## 5. Princípios de implementação

- Python e Jupyter como ambiente principal.
- Modelos transparentes e resultados reproduzíveis.
- Código reutilizável separado dos experimentos didáticos.
- Simulação sintética com parâmetros conhecidos.
- Interfaces padronizadas para modelos e detectores.
- Testes automatizados.
- Separação entre pesquisa, calibração e avaliação fora da amostra.
- Sem obrigação de adotar uma única família de modelos.

## 6. Critérios de sucesso

Ao final, o repositório deverá conter:

- Biblioteca de modelos e interfaces comuns.
- Motor de simulação de trajetórias futuras.
- Pipeline de monitoramento online.
- Mecanismo de avaliação de riscos e alertas.
- Framework de backtesting temporal.
- Notebooks demonstrando cada capacidade.
- Exemplo completo com uma série univariada.

## 7. Limitações e premissas

A detecção de uma mudança estrutural não garante sua antecipação antes de existirem sinais observáveis.

Probabilidades produzidas pelo sistema deverão ser validadas empiricamente.

Simular uma intervenção hipotética não equivale a estimar seu efeito causal.

O escopo atual não inclui modelos globais multivariados, reconciliação MinT, recomendações
automáticas de intervenção ou execução autônoma de ações. A composição compartilha caminhos e
dependência residual, mas níveis agregados ainda não corrigem a média das folhas.

## 8. Estratégia de desenvolvimento

O produto será desenvolvido incrementalmente a partir dos notebooks de aprendizagem. Cada conceito validado poderá gerar uma implementação reutilizável, um conjunto de testes e uma atualização da documentação arquitetural.

## 9. Cobertura implementada na trilha

- caracterização de estacionariedade, memória, resíduos e volatilidade: Notebooks 01–06;
- modelos AR, MA, ARMA, ARIMA e GARCH: Notebooks 02–06;
- simulação Browniana, GBM, choques e Monte Carlo: Notebooks 01, 05 e 06;
- forecasting probabilístico e avaliação rolling-origin: Notebooks 01, 02, 03, 04 e 07;
- contrato de dados, registry, seleção, artefato versionado e monitoramento prequential: Notebook 08;
- surpresa padronizada, EWMA demonstrativa e probabilidade de exceder limite: Notebook 08.
- estado-espaço linear gaussiano, nível local, filtro/smoother de Kalman e gaps: Notebook 09.
- Markov switching gaussiano, probabilidades filtradas/suavizadas e forecast de regimes: Notebook 10.
- change-point offline, CUSUM retrospectivo, segmentação penalizada e avaliação: Notebook 11.
- CUSUM/EWMA sequenciais, ARL, calibração de limiares e atraso de detecção: Notebook 12.
- integração de artefatos imutáveis, monitor causal, risco e política auditável: Notebook 13.
- execução guiada do ciclo completo a partir de uma `pandas.Series`, com política explicável de
  geração de candidatos: Notebook 14.
- composição de séries previstas/conhecidas, cópula residual, covariância e árvore coerente:
  Notebook 15.
- ciclo guiado a partir de `dict[str, list[pd.Series]]`, com grupos e total automáticos:
  Notebook 16.

O Notebook 14 transforma o motor univariado em um fluxo guiado; o Notebook 15 adiciona composição
probabilística sem converter os adaptadores em modelos multivariados. Banco transacional, execução
distribuída, controle de acesso, reconciliação e monitoramento composto permanecem passos de
produção. A demonstração EWMA do Notebook 08, isoladamente, não contava como cobertura conceitual
completa desses detectores.
