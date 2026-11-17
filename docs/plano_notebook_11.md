# Plano didático e técnico — Notebook 11

## Objetivo

Construir detectores offline de mudanças de média e mostrar como custo, penalidade, tamanho mínimo
e hipótese nula determinam a segmentação. O capítulo separa explicitamente outlier, regime
recorrente, drift gradual e quebra estrutural.

## Sequência

1. taxonomia dos tipos de mudança;
2. modelo gaussiano por segmentos;
3. custo SSE e cálculo por somas prefixadas;
4. busca exata de uma quebra;
5. CUSUM offline e localização;
6. calibração Monte Carlo sob ausência de mudança;
7. programação dinâmica para múltiplas quebras;
8. efeito de penalidade e tamanho mínimo;
9. perfil de custo e incerteza de localização;
10. avaliação com tolerância e matching um-para-um;
11. limitações sob outliers, caudas pesadas e autocorrelação;
12. exercícios, gabarito e testes rápidos.

## Formulação central

$$Y_t=\mu_j+\varepsilon_t,\qquad \tau_j<t\le\tau_{j+1},$$

$$C(a,b)=\sum_{t=a}^{b-1}(Y_t-\widehat\mu_{a:b})^2,$$

$$F(t)=\min_s\{F(s)+C(s,t)+\beta\}.$$

## Critérios

- busca de uma quebra e segmentação múltipla implementadas explicitamente;
- penalização e tamanho mínimo interpretados como hipóteses;
- calibração sob hipótese nula demonstrada;
- avaliação com tolerância sem múltipla contagem;
- distinção clara entre análise offline e detecção online;
- módulo reutilizável, testes unitários e execução integral em kernel limpo.
