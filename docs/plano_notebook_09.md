# Plano didático e técnico — Notebook 09

## Objetivo

Introduzir modelos lineares gaussianos de estado-espaço e derivar o filtro de Kalman a partir de
distribuições condicionais. O modelo de nível local separa dinâmica latente e ruído de observação,
permitindo comparar estado verdadeiro, estado filtrado, estado suavizado e forecast.

## Sequência didática

1. problema de medição ruidosa e necessidade de estado latente;
2. formulação geral de transição e observação;
3. simulação explícita do nível local;
4. derivação das etapas de previsão e atualização;
5. interpretação da inovação, de sua variância e do ganho de Kalman;
6. experimento interativo com a razão sinal-ruído;
7. verossimilhança das inovações e estimação de `q` e `r`;
8. diagnóstico de inovações padronizadas;
9. tratamento probabilístico de observações ausentes;
10. smoother de Rauch–Tung–Striebel e distinção entre filtragem e suavização;
11. forecast probabilístico e relação com ARIMA(0,1,1);
12. exercícios, gabarito e testes rápidos.

## Formulação central

O laboratório utiliza

$$\alpha_t=\alpha_{t-1}+\eta_t,\quad \eta_t\sim N(0,q),$$

$$y_t=\alpha_t+\varepsilon_t,\quad \varepsilon_t\sim N(0,r),$$

e torna explícitas as recursões $a_t,P_t,v_t,F_t,K_t,m_t,C_t$. A log-verossimilhança é construída
como soma das densidades preditivas de um passo.

## Critérios de conclusão

- funções matemáticas centrais visíveis no notebook;
- comparação com estado latente conhecido e com `statsmodels`;
- demonstração correta de gaps sem interpolação artificial;
- distinção verificável entre variância do estado e da observação;
- exercícios e gabarito;
- módulo reutilizável em `src/time_series_lab/state_space.py`;
- testes unitários e execução integral em kernel limpo.
