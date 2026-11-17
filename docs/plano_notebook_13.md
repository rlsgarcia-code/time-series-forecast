# Plano didático e técnico — Notebook 13

## Objetivo

Integrar contratos de dados, modelos probabilísticos, seleção temporal, artefatos imutáveis,
monitoramento prequential, detectores sequenciais, change-point retrospectivo, risco e política de
alertas em um fluxo univariado end-to-end auditável.

## Fluxo

`dados → contrato → candidatos → backtest → seleção → forecast congelado → realizado → surpresa → detectores → política → auditoria`.

## Invariantes

1. todo forecast declara cutoff anterior ao alvo;
2. seleção usa somente origens anteriores ao monitoramento;
3. artefatos têm identificador por conteúdo e não são sobrescritos;
4. eventos de monitoramento referenciam o `forecast_id` original;
5. detectores são atualizados somente depois da chegada do realizado;
6. cooldown e reset pertencem a uma política versionável;
7. risco marginal, risco de trajetória, anomalia e drift são armazenados separadamente.

## Sequência

1. contrato temporal com timestamps e frequência;
2. registry probabilístico de AR(1) e nível local;
3. backtest e seleção com CRPS/cobertura;
4. criação e persistência de artefato imutável;
5. loop prequential de um passo;
6. CUSUM/EWMA e política de alerta;
7. confirmação retrospectiva por segmentação;
8. risco marginal e conjunto por Monte Carlo;
9. auditoria de invariantes;
10. limites para implantação real;
11. exercícios, gabarito e testes rápidos.

## Critérios

- fluxo completo executado com mudança simulada fora da seleção;
- forecast e evento ligados por identificador verificável;
- armazenamento recusa sobrescrita e detecta adulteração;
- seleção, monitoramento e risco não usam informação futura;
- alertas respeitam cooldown;
- módulo operacional reutilizável e testes automatizados;
- notebook executado integralmente em kernel limpo.
