# Plano didático e técnico — Notebook 10

## Objetivo

Introduzir cadeias de Markov e modelos gaussianos de mudança de regime, tornando explícitas as
probabilidades previstas, filtradas e suavizadas. O capítulo conecta estados discretos ocultos,
verossimilhança preditiva e forecasting por misturas.

## Sequência

1. hipótese de Markov e matriz de transição;
2. distribuição estacionária e duração esperada;
3. simulação de estados e emissões;
4. derivação e implementação do filtro de Hamilton;
5. smoother de Kim e risco de leakage;
6. máxima verossimilhança com restrições parametrizadas;
7. verificação independente com `statsmodels`;
8. forecast de probabilidades de regime e momentos da mistura;
9. probabilidade de excedência sob mistura;
10. experimento interativo de persistência;
11. distinção entre Markov switching e change point;
12. exercícios, gabarito e testes rápidos.

## Formulação central

$$P(S_t=j\mid S_{t-1}=i)=p_{ij},\qquad
Y_t\mid S_t=k\sim N(\mu_k,\sigma_k^2).$$

O algoritmo forward produz $P(S_t=k\mid y_{1:t})$ e a densidade preditiva de cada observação. O
smoother produz $P(S_t=k\mid y_{1:T})$, enquanto o forecast propaga probabilidades por $P^h$.

## Critérios

- formulação e implementação explícitas do filtro e smoother;
- estimação com identificação dos rótulos;
- comparação com implementação independente;
- forecast preservando a mistura de regimes;
- distinção conceitual de change-point;
- módulo reutilizável, testes unitários e notebook executável em kernel limpo.
