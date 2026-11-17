# Plano didático e técnico — Notebook 12

## Objetivo

Construir detectores sequenciais CUSUM e EWMA sobre surpresas prequentialmente válidas, calibrando
limiares por falso alarme, run length e atraso em vez de selecionar parâmetros visualmente.

## Sequência

1. resíduos de forecast padronizados e ordem prequential;
2. inflação do erro ao repetir testes pontuais;
3. derivação do CUSUM por razão de verossimilhança;
4. derivação do EWMA e de sua variância transitória;
5. definições de ARL, falso alarme, detecção, atraso e censura;
6. calibração Monte Carlo sob estabilidade;
7. avaliação congelada sob mudanças alternativas;
8. experimento interativo de sensibilidade;
9. impacto de escala errada, caudas e autocorrelação;
10. política operacional de reset, cooldown e auditoria;
11. exercícios, gabarito e testes rápidos.

## Formulação central

$$C_t^+=\max(0,C_{t-1}^++Z_t-k),\qquad
C_t^-=\min(0,C_{t-1}^-+Z_t+k),$$

$$E_t=\lambda Z_t+(1-\lambda)E_{t-1},$$

$$\operatorname{Var}(E_t)=\frac{\lambda}{2-\lambda}
[1-(1-\lambda)^{2t}].$$

## Critérios

- implementações explícitas e causais dos dois detectores;
- limiares transitórios corretos para EWMA;
- calibração e avaliação em simulações separadas;
- métricas de falso alarme, detecção, atraso e censura;
- demonstração do impacto de dependência residual;
- módulo reutilizável, testes unitários e execução integral em kernel limpo.
