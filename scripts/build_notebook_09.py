from pathlib import Path

from notebook_builder import code, md, setup_cell, write_notebook


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "09_estado_espaco_kalman.ipynb"


CELLS = [
    md(r"""
    # Notebook 09 — Modelos de estado-espaço e filtro de Kalman

    **Separar o estado latente do ruído de observação**

    Até aqui, os modelos relacionavam diretamente observações passadas e futuras. Agora
    introduziremos uma quantidade que evolui no tempo, mas não é observada sem erro. O filtro de
    Kalman atualizará a distribuição desse estado cada vez que uma nova observação chegar.
    """),
    md(r"""
    ## Público, pré-requisitos e objetivos

    **Pré-requisitos:** distribuição normal, esperança/variância condicionais, ARIMA e forecast
    probabilístico.

    Ao final, você será capaz de:

    1. distinguir estado latente, observação, ruído de estado e ruído de medição;
    2. escrever um modelo linear gaussiano em forma de estado-espaço;
    3. derivar as etapas de previsão e atualização do filtro de Kalman;
    4. interpretar inovação, variância da inovação e ganho de Kalman;
    5. estimar variâncias por máxima verossimilhança das inovações;
    6. diferenciar filtragem, suavização e previsão;
    7. lidar corretamente com observações ausentes.
    """),
    md(r"""
    ## 1. Por que introduzir um estado latente?

    Considere um sensor que mede uma temperatura verdadeira. A temperatura muda gradualmente,
    enquanto o instrumento adiciona erro. O valor observado mistura dois mecanismos:

    - **dinâmica do estado:** quanto a temperatura verdadeira pode mudar;
    - **ruído de observação:** quão impreciso é o instrumento.

    Suavizar uma série por média móvel não separa probabilisticamente esses mecanismos. O modelo de
    estado-espaço especifica ambos e mantém uma distribuição para aquilo que não observamos.
    """),
    md(r"""
    ## 2. Formulação matemática geral

    Um modelo linear gaussiano possui uma equação de transição e uma de observação:

    $$\alpha_t=T_t\alpha_{t-1}+R_t\eta_t,
    \qquad \eta_t\sim N(0,Q_t),$$

    $$y_t=Z_t\alpha_t+\varepsilon_t,
    \qquad \varepsilon_t\sim N(0,H_t).$$

    Aqui, $\alpha_t$ é o vetor de estado latente; $y_t$ é observado; $T_t$ propaga o estado;
    $Z_t$ traduz estado em observação; $Q_t$ mede incerteza da dinâmica; e $H_t$ mede erro de
    observação. Assumimos $\eta_t$ e $\varepsilon_s$ independentes entre si e ao longo do tempo.

    O caso escalar de **nível local** será nosso laboratório:

    $$\alpha_t=\alpha_{t-1}+\eta_t,\qquad \eta_t\sim N(0,q),$$
    $$y_t=\alpha_t+\varepsilon_t,\qquad \varepsilon_t\sim N(0,r).$$

    O quociente $q/r$ controla a flexibilidade: $q$ grande permite que o estado se mova; $r$ grande
    faz o filtro desconfiar das observações individuais.
    """),
    setup_cell("from dataclasses import dataclass\nfrom scipy.optimize import minimize\nimport ipywidgets as widgets\nfrom statsmodels.tsa.statespace.structural import UnobservedComponents"),
    md(r"""
    ## 3. Simulação: verdade latente e medição

    A recorrência abaixo implementa literalmente as duas equações. Em dados reais vemos apenas
    `observation`; `state` existe aqui porque a simulação nos permite verificar o filtro contra a
    verdade conhecida.
    """),
    code("""
    def simulate_local_level(n, *, initial_state=0.0, q, r, seed=None):
        '''Simula alpha_t=alpha_(t-1)+eta_t e y_t=alpha_t+epsilon_t.'''
        if n < 1 or q < 0 or r < 0:
            raise ValueError("n positivo e variâncias não negativas são necessários")
        rng = np.random.default_rng(seed)
        state = np.empty(n)
        state[0] = initial_state + rng.normal(0, np.sqrt(q))
        for t in range(1, n):
            state[t] = state[t-1] + rng.normal(0, np.sqrt(q))
        observation = state + rng.normal(0, np.sqrt(r), n)
        return state, observation

    TRUE = {"q": 0.06, "r": 0.80}
    state, observation = simulate_local_level(300, **TRUE, seed=901)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(observation, color=COLORS["gray"], alpha=.55, label=r"observação $y_t$")
    ax.plot(state, color="black", linewidth=2, label=r"estado $\alpha_t$")
    ax.set(title="Um estado não observado através de medições ruidosas", xlabel="tempo")
    ax.legend(); plt.show()
    """),
    md(r"""
    ## 4. Filtro de Kalman: previsão antes da observação

    Suponha que, após processar $y_{1:t-1}$,

    $$\alpha_{t-1}\mid\mathcal F_{t-1}\sim N(m_{t-1},C_{t-1}).$$

    Pela equação de transição do nível local,

    $$a_t=E_{t-1}[\alpha_t]=m_{t-1},$$
    $$P_t=\operatorname{Var}_{t-1}(\alpha_t)=C_{t-1}+q.$$

    $a_t$ e $P_t$ são a média e a variância **preditas antes de ver $y_t$**. Para a observação,

    $$\widehat y_{t\mid t-1}=a_t,\qquad
    v_t=y_t-a_t,
    \qquad F_t=P_t+r.$$

    $v_t$ é a inovação: a informação realmente nova trazida por $y_t$. Sob o modelo correto,
    $v_t\mid\mathcal F_{t-1}\sim N(0,F_t)$.
    """),
    md(r"""
    ## 5. Atualização após a observação

    Como $(\alpha_t,y_t)$ é conjuntamente normal, condicionar em $y_t$ produz

    $$K_t=\frac{P_t}{F_t}=\frac{P_t}{P_t+r},$$
    $$m_t=a_t+K_tv_t,$$
    $$C_t=(1-K_t)P_t.$$

    O ganho $K_t\in[0,1]$ determina quanto da surpresa entra na estimativa do estado. Ele não é um
    parâmetro arbitrário: é consequência da fórmula de condicionamento da normal multivariada.
    Se $r\to0$, $K_t\to1$ e confiamos na observação; se $r$ é grande frente a $P_t$, o ganho cai.
    """),
    code("""
    @dataclass(frozen=True)
    class KalmanResult:
        predicted_mean: np.ndarray
        predicted_variance: np.ndarray
        innovation: np.ndarray
        innovation_variance: np.ndarray
        gain: np.ndarray
        filtered_mean: np.ndarray
        filtered_variance: np.ndarray
        loglikelihood: float

    def kalman_filter_local_level(y, *, initial_mean, initial_variance, q, r):
        '''Filtro de Kalman escalar; np.nan significa observação ausente.'''
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or initial_variance < 0 or q < 0 or r <= 0:
            raise ValueError("entrada ou variâncias inválidas")
        n = len(y)
        a=np.empty(n);P=np.empty(n);v=np.full(n,np.nan);F=np.full(n,np.nan)
        K=np.full(n,np.nan);m=np.empty(n);C=np.empty(n);loglik=0.0
        previous_mean, previous_variance = initial_mean, initial_variance
        for t in range(n):
            # Previsão: p(alpha_t | y_1:t-1)
            a[t] = previous_mean
            P[t] = previous_variance + q
            if np.isnan(y[t]):
                # Sem medição, previsão vira filtragem sem atualização.
                m[t], C[t] = a[t], P[t]
            else:
                # Atualização: p(alpha_t | y_1:t)
                v[t] = y[t] - a[t]
                F[t] = P[t] + r
                K[t] = P[t] / F[t]
                m[t] = a[t] + K[t] * v[t]
                C[t] = (1-K[t]) * P[t]
                loglik += -.5*(np.log(2*np.pi)+np.log(F[t])+v[t]**2/F[t])
            previous_mean, previous_variance = m[t], C[t]
        return KalmanResult(a,P,v,F,K,m,C,float(loglik))

    filtered = kalman_filter_local_level(
        observation, initial_mean=0, initial_variance=10, **TRUE
    )
    """),
    code("""
    filtered_std = np.sqrt(filtered.filtered_variance)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(observation, color=COLORS["gray"], alpha=.35, label="observado")
    ax.plot(state, color="black", linewidth=2, label="estado verdadeiro")
    ax.plot(filtered.filtered_mean, color=COLORS["blue"], label="estado filtrado")
    ax.fill_between(np.arange(len(state)), filtered.filtered_mean-1.96*filtered_std,
        filtered.filtered_mean+1.96*filtered_std, color=COLORS["blue"], alpha=.15,
        label="intervalo filtrado 95%")
    ax.legend(); ax.set(title="Filtragem usa apenas presente e passado", xlabel="tempo")
    plt.show()

    pd.Series({
        "RMSE da observação": np.sqrt(np.mean((observation-state)**2)),
        "RMSE do estado filtrado": np.sqrt(np.mean((filtered.filtered_mean-state)**2)),
        "ganho final": filtered.gain[-1],
    }).round(4)
    """),
    md(r"""
    **Leitura:** a faixa azul quantifica incerteza sobre o estado, não sobre uma nova observação.
    Uma observação futura inclui também $r$ e terá intervalo mais largo. Nos primeiros instantes, o
    ganho muda porque a incerteza inicial ainda está sendo aprendida; depois converge para um valor
    estável no modelo homogêneo.
    """),
    md(r"""
    ## 6. Experimento interativo: razão sinal-ruído

    Os controles abaixo reutilizam os mesmos choques padronizados. Assim, somente $q$ e $r$ mudam.
    Observe que aumentar $q$ torna o estado filtrado mais responsivo, enquanto aumentar $r$ torna a
    curva mais conservadora em relação a movimentos isolados.
    """),
    code("""
    base_rng = np.random.default_rng(902)
    base_eta = base_rng.normal(size=220)
    base_epsilon = base_rng.normal(size=220)

    def explore_variances(log10_q=-1.2, log10_r=-0.1):
        q, r = 10**log10_q, 10**log10_r
        latent = np.cumsum(np.sqrt(q)*base_eta)
        measured = latent + np.sqrt(r)*base_epsilon
        result = kalman_filter_local_level(measured, initial_mean=0,
            initial_variance=10, q=q, r=r)
        fig, axes = plt.subplots(1,2,figsize=(13,4))
        axes[0].plot(measured,color=COLORS["gray"],alpha=.35)
        axes[0].plot(latent,color="black",label="estado")
        axes[0].plot(result.filtered_mean,color=COLORS["blue"],label="filtro")
        axes[0].legend();axes[0].set_title(f"q={q:.3f}, r={r:.3f}")
        axes[1].plot(result.gain,color=COLORS["orange"])
        axes[1].set(title="Ganho de Kalman",ylim=(0,1),xlabel="tempo")
        plt.tight_layout();plt.show()

    widgets.interact(explore_variances,
        log10_q=widgets.FloatSlider(min=-3,max=.2,step=.1,value=-1.2,description="log10 q"),
        log10_r=widgets.FloatSlider(min=-3,max=.5,step=.1,value=-.1,description="log10 r"));
    """),
    md(r"""
    ## 7. Verossimilhança pelas inovações

    O filtro de Kalman não serve apenas para reconstruir estados. Ele também decompõe a densidade
    conjunta em previsões de um passo:

    $$p(y_{1:T}\mid q,r)=\prod_{t=1}^{T}p(y_t\mid y_{1:t-1},q,r).$$

    Como $v_t\sim N(0,F_t)$, a log-verossimilhança é

    $$\ell(q,r)=-\frac12\sum_{t=1}^{T}
    \left[\log(2\pi)+\log F_t+\frac{v_t^2}{F_t}\right].$$

    Otimizaremos em $(\log q,\log r)$ para garantir variâncias positivas. Isso não elimina
    incerteza paramétrica: produz estimativas plug-in que devem ser diagnosticadas fora da amostra.
    """),
    code("""
    def fit_local_level_mle(y, initial_mean=0, initial_variance=10):
        def objective(log_variances):
            q, r = np.exp(log_variances)
            result = kalman_filter_local_level(y, initial_mean=initial_mean,
                initial_variance=initial_variance, q=q, r=r)
            return -result.loglikelihood
        optimization = minimize(objective, np.log([.1, .5]), method="L-BFGS-B")
        q_hat, r_hat = np.exp(optimization.x)
        result = kalman_filter_local_level(y, initial_mean=initial_mean,
            initial_variance=initial_variance, q=q_hat, r=r_hat)
        return q_hat, r_hat, result, optimization

    q_hat, r_hat, fitted_filter, optimization = fit_local_level_mle(observation)
    pd.DataFrame({"parâmetro":["q: variância do estado","r: variância da observação"],
        "verdadeiro":[TRUE["q"],TRUE["r"]],"estimado":[q_hat,r_hat]}).round(4)
    """),
    md(r"""
    ## 8. Diagnóstico pelas inovações padronizadas

    Se o modelo está bem especificado,

    $$e_t=\frac{v_t}{\sqrt{F_t}}$$

    deve se comportar aproximadamente como ruído branco $N(0,1)$. Autocorrelação sugere dinâmica
    ausente; variância diferente de 1 sugere escala errada; caudas pesadas sugerem distribuição de
    observação inadequada. O diagnóstico é feito sobre inovações de um passo, não sobre estados
    suavizados.
    """),
    code("""
    standardized = fitted_filter.innovation / np.sqrt(fitted_filter.innovation_variance)
    standardized = standardized[np.isfinite(standardized)]
    from statsmodels.tsa.stattools import acf
    lags=np.arange(21);acf_values=acf(standardized,nlags=20,fft=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].plot(standardized,color=COLORS["blue"]);axes[0].set_title("Inovações padronizadas")
    axes[1].hist(standardized,bins=30,density=True,color=COLORS["blue"],alpha=.7)
    grid=np.linspace(-4,4,300);axes[1].plot(grid,stats.norm.pdf(grid),color=COLORS["orange"])
    axes[1].set_title("Distribuição")
    axes[2].vlines(lags,0,acf_values,color=COLORS["purple"],lw=3);axes[2].axhline(0,color="black")
    axes[2].set_title("ACF das inovações")
    plt.tight_layout();plt.show()
    pd.Series({"média":standardized.mean(),"desvio":standardized.std(ddof=1)}).round(3)
    """),
    md(r"""
    ## 9. Dados ausentes: propagar sem inventar

    Se $y_t$ está ausente, não existe inovação. O filtro executa apenas a previsão:
    $m_t=a_t$ e $C_t=P_t$. A incerteza cresce por $q$ até uma nova medição chegar. Interpolar antes
    do filtro criaria informação artificial e poderia reduzir indevidamente a incerteza.
    """),
    code("""
    with_gap = observation.copy(); with_gap[130:155] = np.nan
    gap_filter = kalman_filter_local_level(with_gap, initial_mean=0,
        initial_variance=10, q=q_hat, r=r_hat)
    fig,ax=plt.subplots(figsize=(12,4.5))
    ax.plot(observation,color=COLORS["gray"],alpha=.25,label="observações completas")
    ax.plot(with_gap,color="black",lw=1,label="observado pelo filtro")
    ax.plot(gap_filter.filtered_mean,color=COLORS["blue"],label="estado filtrado")
    band=1.96*np.sqrt(gap_filter.filtered_variance)
    ax.fill_between(np.arange(len(with_gap)),gap_filter.filtered_mean-band,
        gap_filter.filtered_mean+band,color=COLORS["blue"],alpha=.15)
    ax.axvspan(130,154,color=COLORS["orange"],alpha=.12,label="gap")
    ax.legend();ax.set_title("No gap, a distribuição é propagada sem atualização");plt.show()
    """),
    md(r"""
    ## 10. Filtragem não é suavização

    - $m_t=E[\alpha_t\mid y_{1:t}]$ é **filtrado** e pode ser calculado online.
    - $\widetilde m_t=E[\alpha_t\mid y_{1:T}]$ é **suavizado** e usa observações futuras.

    O smoother de Rauch–Tung–Striebel percorre a amostra de trás para frente:

    $$J_t=\frac{C_t}{P_{t+1}},$$
    $$\widetilde m_t=m_t+J_t(\widetilde m_{t+1}-a_{t+1}),$$
    $$\widetilde C_t=C_t+J_t^2(\widetilde C_{t+1}-P_{t+1}).$$

    Suavização é adequada para reconstrução histórica, mas não pode ser usada como se estivesse
    disponível no instante $t$ durante um backtest.
    """),
    code("""
    def kalman_smoother_local_level(result):
        smoothed_mean=result.filtered_mean.copy()
        smoothed_variance=result.filtered_variance.copy()
        for t in range(len(smoothed_mean)-2,-1,-1):
            J=result.filtered_variance[t]/result.predicted_variance[t+1]
            smoothed_mean[t]=result.filtered_mean[t]+J*(
                smoothed_mean[t+1]-result.predicted_mean[t+1])
            smoothed_variance[t]=result.filtered_variance[t]+J**2*(
                smoothed_variance[t+1]-result.predicted_variance[t+1])
        return smoothed_mean,smoothed_variance

    smoothed_mean,smoothed_variance=kalman_smoother_local_level(fitted_filter)
    fig,ax=plt.subplots(figsize=(12,5))
    ax.plot(state,color="black",lw=2,label="estado verdadeiro")
    ax.plot(fitted_filter.filtered_mean,color=COLORS["blue"],alpha=.7,label="filtrado")
    ax.plot(smoothed_mean,color=COLORS["orange"],label="suavizado")
    ax.legend();ax.set_title("Smoother usa toda a amostra para revisar o passado");plt.show()
    pd.Series({"RMSE filtrado":np.sqrt(np.mean((fitted_filter.filtered_mean-state)**2)),
        "RMSE suavizado":np.sqrt(np.mean((smoothed_mean-state)**2))}).round(4)
    """),
    md(r"""
    ## 11. Forecast probabilístico

    Após observar $y_{1:T}$, temos $\alpha_T\mid\mathcal F_T\sim N(m_T,C_T)$. Para $h\ge1$,

    $$\alpha_{T+h}\mid\mathcal F_T\sim N(m_T,C_T+hq),$$

    $$y_{T+h}\mid\mathcal F_T\sim N(m_T,C_T+hq+r).$$

    A média é constante porque o nível local não possui drift; a variância cresce porque novos
    choques de estado se acumulam. A variância de observação inclui ainda $r$. Confundir as duas
    distribuições produz intervalos estreitos demais para medições futuras.
    """),
    code("""
    def forecast_local_level(result,horizon,*,q,r):
        h=np.arange(1,horizon+1)
        mean=np.repeat(result.filtered_mean[-1],horizon)
        state_variance=result.filtered_variance[-1]+h*q
        observation_variance=state_variance+r
        return h,mean,state_variance,observation_variance

    h,mean,state_var,obs_var=forecast_local_level(fitted_filter,24,q=q_hat,r=r_hat)
    future_index=len(observation)+h-1
    fig,ax=plt.subplots(figsize=(12,5))
    ax.plot(np.arange(len(observation)-80,len(observation)),observation[-80:],color="black",label="histórico")
    ax.plot(future_index,mean,color=COLORS["orange"],label="média preditiva")
    ax.fill_between(future_index,mean-1.96*np.sqrt(obs_var),mean+1.96*np.sqrt(obs_var),
        color=COLORS["orange"],alpha=.2,label="observação futura 95%")
    ax.legend();ax.set(title="Forecast do modelo de nível local",xlabel="tempo");plt.show()
    """),
    md(r"""
    ## 12. Relação com ARIMA e validação independente

    O nível local é estrutural: descreve separadamente estado e medição. Eliminando o estado,
    $Delta y_t=\eta_t+\varepsilon_t-\varepsilon_{t-1}$ possui uma representação
    ARIMA$(0,1,1)$. A correspondência entre parâmetros não é simplesmente $\theta=-1$; o
    coeficiente MA depende da razão $q/r$.

    Modelos estrutural e reduzido podem produzir forecasts semelhantes, mas respondem perguntas
    diferentes. Estado-espaço permite estados interpretáveis, componentes, dados ausentes e
    extensões multivariadas. Abaixo comparamos nossa implementação com `statsmodels`.
    """),
    code("""
    reference = UnobservedComponents(observation, level="local level").fit(disp=False)
    pd.DataFrame({
        "quantidade":["q estimado manualmente","r estimado manualmente",
            "q statsmodels","r statsmodels","loglik manual","loglik statsmodels"],
        "valor":[q_hat,r_hat,reference.params[1],reference.params[0],
            fitted_filter.loglikelihood,reference.llf]
    }).round(4)
    """),
    md(r"""
    ## 13. Exercícios

    1. Derive $F_t=P_t+r$ usando independência entre estado e erro de observação.
    2. Explique por que $K_t$ aumenta quando $r$ diminui.
    3. Fixe $r=1$ e compare filtros com $q=0{,}001$ e $q=1$ usando os mesmos choques.
    4. Introduza um gap de 50 períodos e descreva a evolução de $C_t$.
    5. Mostre por que usar o estado suavizado em uma avaliação online causa leakage.
    6. Acrescente drift $d$ à equação de estado e derive a média do forecast em $h$.
    7. Compare os diagnósticos de inovação para um modelo com $q$ propositalmente muito pequeno.
    """),
    md(r"""
    <details><summary><strong>Gabarito orientativo</strong></summary>

    1. $y_t=\alpha_t+\varepsilon_t$ e as parcelas são independentes condicionalmente ao passado,
       logo as variâncias somam.
    2. $K_t=P_t/(P_t+r)$: reduzir o denominador aumenta o peso da medição.
    3. Com $q$ pequeno o estado quase não se move; com $q$ grande o filtro acompanha rapidamente.
    4. Sem atualização, $C_t=C_{t-1}+q$, portanto a incerteza cresce linearmente.
    5. $\widetilde m_t$ usa $y_{t+1:T}$, indisponível quando a previsão em $t$ seria emitida.
    6. $\alpha_t=\alpha_{t-1}+d+\eta_t$ implica $E_T[\alpha_{T+h}]=m_T+hd$.
    7. Surgem inovações persistentes porque o estado modelado reage devagar demais.

    </details>
    """),
    code("""
    # Testes rápidos das identidades centrais.
    assert optimization.success
    assert q_hat > 0 and r_hat > 0
    assert np.all((fitted_filter.gain >= 0) & (fitted_filter.gain <= 1))
    assert np.all(np.diff(state_var) > 0)
    np.testing.assert_allclose(obs_var-state_var,r_hat)
    assert np.isnan(gap_filter.gain[140])
    assert gap_filter.filtered_mean[140] == gap_filter.predicted_mean[140]
    np.testing.assert_allclose(smoothed_mean[-1],fitted_filter.filtered_mean[-1])
    assert abs(reference.params[0]-r_hat) < .05
    assert abs(reference.params[1]-q_hat) < .05
    print("Testes rápidos do Notebook 09 passaram.")
    """),
    md(r"""
    ## Síntese e próximos passos

    O filtro de Kalman é uma sequência de distribuições condicionais: prever, observar a inovação
    e atualizar. Ele separa incerteza de estado e observação, lida naturalmente com gaps e fornece
    a verossimilhança usada para aprender hiperparâmetros.

    **Armadilhas principais:** confundir filtragem com suavização; interpretar estado como verdade
    observada; ignorar incerteza dos parâmetros; usar $C_t$ como variância de uma nova observação;
    e avaliar online com informação futura.

    O próximo notebook permitirá que a dinâmica mude entre estados discretos por meio de cadeias
    de Markov, introduzindo modelos de regimes e probabilidades filtradas de cada regime.
    """),
]


if __name__ == "__main__":
    write_notebook(PATH, CELLS, 9)
