# Avaliação do desdobramento dos Notebooks 03 e 04

## Decisão

Manter os Notebooks 03 e 04 como capítulos únicos na trilha principal.

O Notebook 03 possui uma pergunta coerente — como valores e choques defasados formam AR(p), MA(q)
e ARMA — e já contém formulação por polinômios, raízes, resposta a impulso, ACF/PACF, seleção,
diagnóstico e backtest. Separá-lo agora repetiria definições e exigiria renumerar toda a trilha.

O Notebook 04 possui outra pergunta coerente — como diagnosticar e modelar não estacionariedade —
ligando tendência, raiz unitária, diferenciação, ADF, ARIMA e retorno ao nível original. A
integração entre diferença e forecast em nível é justamente o resultado pedagógico do capítulo.

## Evidência considerada

- Notebook 03: 29 células, com 13 células de código;
- Notebook 04: 25 células, com 11 células de código;
- ambos possuem espinha matemática, exercícios, gabarito, testes e validação em kernel limpo;
- nenhum deles excede o tamanho dos capítulos avançados 09–13;
- a principal lacuna identificada era a ausência de Kalman, regimes e detecção, não a inexistência
  de ARMA/ARIMA.

## Extensões avançadas possíveis

Se houver necessidade pedagógica posterior, criar complementos sem duplicar a trilha principal:

- **03A — Inferência ARMA:** equações de Yule–Walker, likelihood exata/condicional, matriz de
  informação, incerteza dos parâmetros e previsão por representação psi;
- **04A — Raízes unitárias avançadas:** potência e tamanho dos testes, KPSS, quebras confundidas com
  raiz unitária e simulações locais a 1;
- **04B — Cointegração e correção de erros:** somente quando o escopo avançar para séries
  multivariadas.

Esses complementos são aprofundamentos opcionais. Não são necessários para a coerência da solução
univariada 01–13 entregue agora.
