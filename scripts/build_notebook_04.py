from pathlib import Path

from notebook_builder import code, md, setup_cell, write_notebook

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "04_nao_estacionariedade_arima.ipynb"

CELLS = [
    md(r"""
    # Notebook 04 — Não estacionariedade e ARIMA

    **Tendência, raiz unitária, diferenciação e retorno à escala original**

    ARMA pressupõe uma estrutura estável. Aqui aprendemos a reconhecer quando o nível não é
    estacionário, aplicar diferenças sem destruir a interpretação e prever novamente em nível.
    """),
    md(r"""
    ## Objetivos e roteiro

    Distinguir tendência determinística de raiz unitária; interpretar o teste ADF; compreender
    $\Delta=(1-B)$; escolher `d` com parcimônia; ajustar ARIMA(p,d,q); inverter previsões e evitar
    regressão espúria e sobrediferenciação.
    """),
    md(r"""
    ## Mapa conceitual: o que exatamente pode não ser estacionário?

    Estacionariedade fraca exige três propriedades estáveis: média constante,
    variância constante e covariância dependente apenas da distância entre tempos. Uma série pode
    violar uma ou várias delas.

    | Mecanismo | O que muda? | Efeito de um choque | Tratamento típico |
    |---|---|---|---|
    | tendência determinística | média conhecida como função de $t$ | temporário ao redor da tendência | regressão/detrending |
    | raiz unitária | nível estocástico | permanente | diferenciação |
    | variância condicional | escala dos erros | clusters de amplitude | ARCH/GARCH |
    | quebra estrutural | parâmetro em uma data/regime | depende do novo regime | detecção/modelo de regime |

    Tendência visual não identifica o mecanismo. A pergunta decisiva é: depois de um choque, a
    série retorna a uma trajetória previamente definida ou passa a carregar um novo nível?
    """),
    md(r"""
    ## Espinha matemática: estacionariedade, integração e ARIMA

    Um processo é fracamente estacionário quando, para todo $t$ e lag $k$,

    $$E[X_t]=\mu,\qquad \operatorname{Var}(X_t)=\gamma(0)<\infty,
    \qquad \operatorname{Cov}(X_t,X_{t-k})=\gamma(k).$$

    Dizemos que $X_t$ é integrado de ordem $d$, $X_t\sim I(d)$, se $(1-B)^dX_t$ é estacionário
    e $d$ é o menor inteiro com essa propriedade. Um ARIMA$(p,d,q)$ é

    $$\phi(B)(1-B)^dX_t=c+\theta(B)\varepsilon_t.$$

    Para $d=1$, $\Delta X_t=(1-B)X_t=X_t-X_{t-1}$. A transformação é invertida por

    $$X_{T+h}=X_T+\sum_{j=1}^{h}\Delta X_{T+j}.$$

    Por isso, erros previstos nas diferenças se acumulam e ficam correlacionados no nível. O ADF
    estima

    $$\Delta X_t=\alpha+\beta t+\gamma X_{t-1}
    +\sum_{i=1}^{m}\delta_i\Delta X_{t-i}+u_t$$

    e testa $H_0:\gamma=0$ contra uma alternativa estacionária. Não rejeitar $H_0$ não prova raiz
    unitária; indica apenas que a amostra e a especificação não produziram evidência suficiente
    contra ela.
    """),
    setup_cell("from statsmodels.tsa.arima.model import ARIMA\nfrom statsmodels.tsa.stattools import adfuller, acf"),
    md(r"""
    ## 1. Três séries visualmente crescentes

    Uma subida pode vir de: tendência determinística $a+bt+u_t$, passeio aleatório com drift
    $X_t=X_{t-1}+d+\varepsilon_t$, ou processo estacionário em torno de média. O mecanismo importa
    porque choques são temporários no primeiro caso e permanentes no segundo.

    Na tendência determinística, subtrair $a+bt$ revela $u_t$ estacionário. No passeio aleatório,
    não existe uma linha fixa à qual retornar: cada inovação redefine toda a trajetória futura.
    Duas amostras do mesmo passeio podem terminar em níveis muito diferentes mesmo com o mesmo drift.
    """),
    code("""
    def simulate_random_walk(n, *, drift=0.0, sigma=1.0, seed=None):
        rng = np.random.default_rng(seed)
        increments = drift + rng.normal(0, sigma, n)
        return np.r_[0.0, np.cumsum(increments)]

    rng = np.random.default_rng(401)
    n = 350
    time = np.arange(n)
    trend_stationary = 2 + 0.04*time + rng.normal(0, 1, n)
    random_walk = simulate_random_walk(n-1, drift=.04, sigma=1, seed=402)
    stationary = np.empty(n); stationary[0] = 2
    noise = rng.normal(0, 1, n)
    for t in range(1, n): stationary[t] = 2 + .75*(stationary[t-1]-2) + noise[t]

    fig, axes = plt.subplots(3,1,figsize=(11,9),sharex=True)
    for ax, values, title in zip(axes, [trend_stationary, random_walk, stationary],
        ["Tendência determinística", "Passeio aleatório com drift", "AR(1) estacionário"]):
        ax.plot(values, color=COLORS["blue"]); ax.set_title(title)
    axes[-1].set_xlabel("tempo"); plt.tight_layout(); plt.show()
    """),
    md(r"""
    ## 2. Teste ADF: hipótese e limitações

    O ADF testa $H_0$: raiz unitária. P-valor pequeno rejeita raiz unitária sob a especificação de
    constante/tendência e número de lags escolhidos. P-valor grande não prova raiz unitária: o teste
    tem baixa potência em amostras curtas e contra alternativas muito persistentes.

    O teste estima uma regressão do tipo

    $$\Delta X_t=\alpha+\beta t+\gamma X_{t-1}+
    \sum_{i=1}^{p}\delta_i\Delta X_{t-i}+u_t.$$

    A hipótese de raiz unitária é $H_0:\gamma=0$; rejeitar em favor de $\gamma<0$ sugere reversão.
    Os termos $\alpha$ e $\beta t$ não são cosméticos: omitir uma tendência determinística ou
    incluir uma desnecessária muda a distribuição do teste e sua potência. Os lags das diferenças
    servem para tornar $u_t$ aproximadamente não autocorrelacionado.
    """),
    code("""
    def adf_summary(values, regression="c"):
        statistic, pvalue, used_lag, nobs, critical, _ = adfuller(values, regression=regression, autolag="AIC")
        return {"estatística": statistic, "p-valor": pvalue, "lags": used_lag, "n": nobs, "crítico 5%": critical["5%"]}

    pd.DataFrame({
        "tendência (com termo de tendência)": adf_summary(trend_stationary, regression="ct"),
        "passeio aleatório": adf_summary(random_walk, regression="c"),
        "AR(1)": adf_summary(stationary, regression="c"),
    }).T.round(4)
    """),
    md(r"""
    ## 3. Diferenciação

    O operador de defasagem satisfaz $BX_t=X_{t-1}$. A primeira diferença é
    $\Delta X_t=(1-B)X_t=X_t-X_{t-1}$. Para passeio aleatório com drift,
    $\Delta X_t=d+\varepsilon_t$, que é estacionário.

    Diferenciar remove uma unidade de integração, mas também muda a pergunta: modelamos variações,
    não níveis.

    **Exemplo:** para níveis `[10, 12, 11, 14]`, a primeira diferença é `[2, -1, 3]`. Conhecendo
    o último nível 14 e futuras diferenças `[1, 0.5, -2]`, reconstruímos `[15, 15.5, 13.5]` por
    soma cumulativa. A transformação perde o nível inicial; por isso ele precisa ser preservado
    para inverter previsões.
    """),
    code("""
    def difference(values, order=1):
        '''Aplica (1-B)^order.'''
        return np.diff(np.asarray(values), n=order)

    rw_diff = difference(random_walk)
    fig, axes = plt.subplots(2,1,figsize=(11,7),sharex=False)
    axes[0].plot(random_walk, color=COLORS["orange"]); axes[0].set_title("Nível não estacionário")
    axes[1].plot(rw_diff, color=COLORS["green"]); axes[1].axhline(.04,color="black",ls="--"); axes[1].set_title("Primeira diferença")
    plt.tight_layout(); plt.show()
    pd.DataFrame({"nível": adf_summary(random_walk), "diferença": adf_summary(rw_diff)}).T.round(4)
    """),
    md(r"""
    ### 3.1 Sobrediferenciação

    Aplicar diferenças demais introduz dependência artificial e aumenta ruído. Um sinal comum é
    forte autocorrelação negativa no lag 1 após diferenciar uma série já estacionária.
    """),
    code("""
    stationary_diff = difference(stationary)
    fig, ax = plt.subplots(figsize=(9,4))
    lags = np.arange(16)
    ax.vlines(lags, 0, acf(stationary_diff, nlags=15), color=COLORS["red"], linewidth=3)
    ax.axhline(0,color="black",lw=.8); ax.set(title="ACF após diferenciar um AR(1) estacionário",xlabel="lag")
    plt.show()
    """),
    md(r"""
    ## 4. ARIMA(p,d,q)

    ARIMA aplica ARMA à série diferenciada:
    $\phi(B)(1-B)^dX_t=\theta(B)\varepsilon_t$. O parâmetro `d` não é escolhido apenas pelo menor
    AIC: combinamos mecanismo, gráficos, teste, ACF e validação.

    | Parâmetro | Atua sobre | Pergunta |
    |---|---|---|
    | $p$ | valores defasados da série diferenciada | quantos estados passados permanecem úteis? |
    | $d$ | transformação $(1-B)^d$ | quantas raízes unitárias remover? |
    | $q$ | inovações defasadas | por quantos períodos um choque direto aparece? |

    ARIMA não significa “diferenciar até o ADF rejeitar”. Cada diferença aumenta ruído, reduz uma
    observação e altera a interpretação. Em `d=1`, um termo de drift nas diferenças implica
    tendência linear esperada nos níveis.
    """),
    code("""
    # Série integrada cujas diferenças seguem ARMA(1,1).
    rng = np.random.default_rng(403)
    diff_process = np.zeros(700)
    eps = rng.normal(0, .8, 700)
    for t in range(1,700): diff_process[t] = .55*diff_process[t-1] + eps[t] + .35*eps[t-1]
    levels = 50 + np.cumsum(.12 + diff_process)
    train, test = levels[:-100], levels[-100:]

    rows = []
    fits = {}
    for order in [(1,0,1),(0,1,0),(1,1,0),(0,1,1),(1,1,1),(2,1,1)]:
        fit = ARIMA(train, order=order, trend="t" if order[1] == 1 else "ct").fit()
        name = f"ARIMA{order}"
        fits[name] = fit
        rows.append({"modelo":name,"AIC":fit.aic,"BIC":fit.bic})
    pd.DataFrame(rows).sort_values("AIC").round(2)
    """),
    md(r"""
    ### 4.1 Forecast e retorno ao nível

    Em uma implementação manual, diferenças previstas são acumuladas:
    $\widehat X_{T+h}=X_T+\sum_{j=1}^{h}\widehat{\Delta X}_{T+j}$. `statsmodels` faz essa inversão
    automaticamente quando `d=1`. A função abaixo deixa a operação explícita.

    A inversão da média é simples, mas intervalos exigem a covariância conjunta das diferenças:
    erros de vários horizontes são correlacionados quando acumulados. Somar separadamente limites
    inferiores e superiores não é, em geral, uma inversão probabilística válida; o modelo ARIMA
    calcula a distribuição integrada de forma coerente.
    """),
    code("""
    def integrate_differences(last_level, predicted_differences):
        return last_level + np.cumsum(predicted_differences)

    demo_diffs = np.array([.5, -.2, .1])
    pd.DataFrame({"diferença prevista":demo_diffs,"nível reconstruído":integrate_differences(10,demo_diffs)})
    """),
    code("""
    evaluation = []
    predictions = {}
    for name, fit in fits.items():
        pred = fit.get_forecast(len(test))
        mean = np.asarray(pred.predicted_mean)
        interval = np.asarray(pred.conf_int())
        predictions[name] = (mean, interval)
        evaluation.append({"modelo":name,"RMSE":np.sqrt(np.mean((test-mean)**2)),
            "cobertura":np.mean((test>=interval[:,0])&(test<=interval[:,1])),
            "largura":np.mean(interval[:,1]-interval[:,0])})
    scores = pd.DataFrame(evaluation).sort_values("RMSE")
    scores.round(3)
    """),
    code("""
    winner = scores.iloc[0]["modelo"]
    mean, interval = predictions[winner]
    future_t = np.arange(len(train),len(levels))
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(np.arange(len(train)-100,len(train)),train[-100:],color="black",label="treino")
    ax.plot(future_t,test,color=COLORS["blue"],label="teste")
    ax.plot(future_t,mean,color=COLORS["orange"],label=winner)
    ax.fill_between(future_t,interval[:,0],interval[:,1],color=COLORS["orange"],alpha=.18)
    ax.legend(); ax.set(title="Forecast ARIMA na escala original",xlabel="tempo")
    plt.show()
    """),
    md(r"""
    ## 5. Regressão espúria

    Duas caminhadas aleatórias independentes podem apresentar correlação e $R^2$ altos porque
    ambas vagam lentamente. Diferenciar ou modelar cointegração (fora do escopo univariado) evita
    interpretar tendências estocásticas comuns como relação econômica real.
    """),
    code("""
    rw_a = simulate_random_walk(499,seed=410)
    rw_b = simulate_random_walk(499,seed=411)
    corr_levels = np.corrcoef(rw_a,rw_b)[0,1]
    corr_diffs = np.corrcoef(np.diff(rw_a),np.diff(rw_b))[0,1]
    pd.Series({"correlação nos níveis":corr_levels,"correlação nas diferenças":corr_diffs}).round(3)
    """),
    md(r"""
    ## 6. Exercícios

    1. Explique a diferença entre remover tendência por regressão e por diferenciação.
    2. Por que p-valor ADF alto não prova raiz unitária?
    3. Diferencie duas vezes um passeio aleatório e examine a ACF.
    4. Reconstrua níveis a partir de `[1,-.5,.2]` e último nível 20.
    5. Compare ARIMA(1,1,1) com random walk no mesmo bloco de teste.
    """),
    md(r"""
    <details><summary><strong>Gabarito comentado</strong></summary>

    1. Regressão remove componente determinística; diferença remove uma raiz unitária e muda a
       variável para variações. 2. O ADF pode ter baixa potência e depende de termos/defasagens.
    3. A segunda diferença tende a criar forte correlação negativa desnecessária. 4. Os níveis são
       21, 20,5 e 20,7. 5. Compare no mesmo cutoff/horizonte; o vencedor não é definido pelo AIC.
    </details>
    """),
    code("""
    exercise_levels = integrate_differences(20,np.array([1,-.5,.2]))
    np.testing.assert_allclose(exercise_levels,[21,20.5,20.7])
    assert adf_summary(rw_diff)["p-valor"] < .05
    assert len(mean) == len(test)
    print("✓ Gabarito numérico e testes rápidos passaram.")
    """),
    md(r"""
    ## Síntese

    Não estacionariedade é uma hipótese sobre o mecanismo, não apenas uma aparência. ADF ajuda,
    mas precisa de especificação e contexto. Diferenciação converte níveis integrados em variações;
    ARIMA modela essas variações e devolve forecasts à escala original. O próximo capítulo passa
    ao tempo contínuo com movimento Browniano, GBM e Itô.
    """),
]

if __name__ == "__main__":
    write_notebook(PATH,CELLS,4)

