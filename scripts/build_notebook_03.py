from pathlib import Path

from notebook_builder import code, md, setup_cell, write_notebook


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "03_arp_maq_arma.ipynb"

CELLS = [
    md(r"""
    # Notebook 03 — AR(p), MA(q) e ARMA

    **Memória de valores, memória de choques e como distingui-las**

    O AR(1) usa apenas o último valor. Neste capítulo ampliamos a memória com vários lags,
    introduzimos choques defasados e usamos ACF/PACF, diagnóstico e backtest para comparar modelos.
    """),
    md(r"""
    ## Objetivos e roteiro

    Ao final, você saberá simular AR(p), MA(q) e ARMA(p,q), interpretar ACF e PACF, ajustar
    candidatos, usar AIC sem confundi-lo com desempenho futuro e selecionar ordens com diagnóstico
    residual e avaliação temporal.

    1. AR(p): memória de valores; 2. MA(q): memória de choques; 3. assinaturas ACF/PACF;
    4. estimação e seleção; 5. forecast fora da amostra; 6. exercícios e testes.
    """),
    md(r"""
    ## Mapa conceitual antes das fórmulas

    Há duas maneiras diferentes de o passado influenciar o presente:

    | Ideia | Pergunta | Termo no modelo |
    |---|---|---|
    | memória de **estado** | valores antigos de $X$ ainda ajudam a prever? | $\phi_iX_{t-i}$ |
    | memória de **choque** | uma inovação antiga ainda afeta a observação? | $\theta_j\varepsilon_{t-j}$ |

    Um termo AR usa algo observável — valores passados da série. Um termo MA usa inovações, que
    são latentes e precisam ser reconstruídas durante o ajuste. “MA” aqui significa *moving
    average de choques*, não a média móvel descritiva usada para suavizar gráficos.

    **Vocabulário:** `p` é o número de lags de $X$; `q`, o número de lags de $\varepsilon$;
    $B$ é o operador de defasagem; estacionariedade controla se choques desaparecem; invertibilidade
    garante que as inovações MA podem ser recuperadas de forma única a partir da série.
    """),
    md(r"""
    ## Espinha matemática: polinômios de defasagem

    Defina o operador $B$ por $BX_t=X_{t-1}$ e

    $$\phi(B)=1-\phi_1B-\cdots-\phi_pB^p,\qquad
    \theta(B)=1+\theta_1B+\cdots+\theta_qB^q.$$

    Um ARMA$(p,q)$ com média zero satisfaz

    $$\phi(B)X_t=\theta(B)\varepsilon_t,\qquad
    \varepsilon_t\overset{iid}{\sim}N(0,\sigma^2).$$

    Se as raízes de $\phi(z)=0$ têm módulo maior que 1, o processo é causal e estacionário:

    $$X_t=\psi(B)\varepsilon_t=\sum_{j=0}^{\infty}\psi_j\varepsilon_{t-j},
    \qquad \psi(B)=\frac{\theta(B)}{\phi(B)}.$$

    Dessa representação seguem

    $$\operatorname{Var}(X_t)=\sigma^2\sum_{j=0}^{\infty}\psi_j^2,\qquad
    \gamma(k)=\sigma^2\sum_{j=0}^{\infty}\psi_j\psi_{j+|k|}.$$

    Se as raízes de $\theta(z)=0$ também estão fora do círculo unitário, o MA é invertível. Para
    previsão,

    $$\widehat X_{T+h\mid T}=E[X_{T+h}\mid\mathcal F_T],\qquad
    \operatorname{Var}(e_{T+h\mid T})=\sigma^2\sum_{j=0}^{h-1}\psi_j^2.$$

    Logo, ACF, PACF, resposta a impulso e variância preditiva são manifestações diferentes dos
    mesmos coeficientes $\psi_j$.
    """),
    setup_cell("from statsmodels.tsa.arima.model import ARIMA\nfrom statsmodels.tsa.stattools import acf, pacf\nfrom statsmodels.stats.diagnostic import acorr_ljungbox"),
    md(r"""
    ## 1. Um simulador único para ARMA

    $$X_t=\sum_{i=1}^{p}\phi_iX_{t-i}+\varepsilon_t+
    \sum_{j=1}^{q}\theta_j\varepsilon_{t-j},\qquad
    \varepsilon_t\sim\mathcal N(0,\sigma^2).$$

    Os coeficientes AR carregam valores passados; os MA carregam inovações passadas. `burnin`
    descarta o início artificial em zero antes de devolver a amostra.
    """),
    code("""
    def simulate_arma(n, *, ar=(), ma=(), sigma=1.0, burnin=300, seed=None):
        '''Simula ARMA(p,q) pela recorrência matemática explícita.'''
        phi, theta = np.asarray(ar), np.asarray(ma)
        p, q = len(phi), len(theta)
        pad = max(p, q)
        total = n + burnin + pad
        rng = np.random.default_rng(seed)
        epsilon = rng.normal(0, sigma, total)
        x = np.zeros(total)
        for t in range(pad, total):
            ar_part = sum(phi[i] * x[t-i-1] for i in range(p))
            ma_part = sum(theta[j] * epsilon[t-j-1] for j in range(q))
            x[t] = ar_part + epsilon[t] + ma_part
        return x[burnin + pad:]
    """),
    md(r"""
    ## 2. AR(p): vários estados passados

    Um AR(2), $X_t=\phi_1X_{t-1}+\phi_2X_{t-2}+\varepsilon_t$, pode produzir decaimento
    monotônico ou oscilações amortecidas. Estacionariedade depende das raízes do polinômio AR,
    não de verificar cada coeficiente isoladamente.
    """),
    md(r"""
    ### 2.1 Como dois lags geram dinâmica

    Imagine um choque unitário em $t=0$ e nenhuma inovação depois. No AR(2), a resposta obedece

    $$\psi_h=\phi_1\psi_{h-1}+\phi_2\psi_{h-2},\quad \psi_0=1.$$

    Portanto, o segundo lag não é apenas “mais memória”: ele altera a forma da resposta. Se
    $\phi_2<0$, o estado de dois períodos atrás empurra na direção oposta e pode gerar oscilação.
    A condição correta de estacionariedade é que as raízes de
    $1-\phi_1z-\phi_2z^2=0$ estejam fora do círculo unitário. Coeficientes individualmente menores
    que 1 não bastam.
    """),
    code("""
    def ar2_impulse_response(phi1, phi2, horizon=12):
        response=np.zeros(horizon+1);response[0]=1
        if horizon>=1: response[1]=phi1
        for h in range(2,horizon+1):response[h]=phi1*response[h-1]+phi2*response[h-2]
        return response

    pd.DataFrame({"h":range(13),"AR(2) oscilatório":ar2_impulse_response(1.2,-.45),
        "AR(2) amortecido":ar2_impulse_response(.5,.2)}).round(3)
    """),
    code("""
    ar1 = simulate_arma(500, ar=[0.75], seed=301)
    ar2 = simulate_arma(500, ar=[1.20, -0.45], seed=301)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(ar1, color=COLORS["blue"]); axes[0].set_title("AR(1): persistência simples")
    axes[1].plot(ar2, color=COLORS["purple"]); axes[1].set_title("AR(2): oscilação amortecida")
    axes[1].set_xlabel("tempo")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ## 3. MA(q): choques com duração finita

    No MA(1), $X_t=\varepsilon_t+\theta_1\varepsilon_{t-1}$. Um choque afeta somente dois
    instantes. Isso é diferente do AR, onde um choque entra no estado e decai indefinidamente.
    O MA é sempre estacionário; a condição de invertibilidade permite representação e estimação
    únicas.
    """),
    md(r"""
    ### 3.1 Por que a memória MA “corta”

    Se $\varepsilon_0=1$ e todos os outros choques são zero, um MA(1) produz $X_0=1$,
    $X_1=\theta_1$ e $X_h=0$ para $h\ge2$. A memória termina por construção. Já no AR, o choque
    altera $X_0$, que entra em $X_1$, depois em $X_2$ e assim por diante.

    Invertibilidade não é o mesmo que estacionariedade. O MA finito já é estacionário; exigir as
    raízes de $1+\theta_1z+\cdots+\theta_qz^q$ fora do círculo unitário escolhe uma representação
    identificável e permite escrever choques como função de observações passadas.
    """),
    code("""
    ma1 = simulate_arma(500, ma=[0.80], seed=302)
    arma11 = simulate_arma(500, ar=[0.65], ma=[0.55], seed=302)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(ma1, color=COLORS["orange"]); axes[0].set_title("MA(1): memória curta de choques")
    axes[1].plot(arma11, color=COLORS["green"]); axes[1].set_title("ARMA(1,1): duas memórias combinadas")
    axes[1].set_xlabel("tempo")
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ## 4. ACF e PACF como assinaturas

    - AR(p): PACF tende a cortar após $p$; ACF decai.
    - MA(q): ACF tende a cortar após $q$; PACF decai.
    - ARMA: ambas geralmente decaem.

    “Cortar” é uma propriedade populacional. Em amostra finita surgem barras espúrias, portanto
    ACF/PACF propõem candidatos; não escolhem o modelo sozinhas.

    Formalmente, $\rho(k)=\mathrm{Corr}(X_t,X_{t-k})$. A PACF no lag $k$ mede a associação entre
    $X_t$ e $X_{t-k}$ **depois de remover linearmente** o efeito dos lags intermediários
    $1,\ldots,k-1$. É essa remoção que faz um AR(p) apresentar corte populacional na PACF.

    As faixas aproximadas $\pm1{,}96/\sqrt n$ avaliam cada lag isoladamente. Olhar dezenas de lags
    aumenta a chance de barras aparentemente significativas por acaso; procure padrão, não um pico
    solitário.
    """),
    code("""
    processes = {"AR(2)": ar2, "MA(1)": ma1, "ARMA(1,1)": arma11}
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    lags = np.arange(21)
    for row, (name, values) in enumerate(processes.items()):
        acf_values = acf(values, nlags=20, fft=True)
        pacf_values = pacf(values, nlags=20, method="ywm")
        axes[row, 0].vlines(lags, 0, acf_values, color=COLORS["blue"], linewidth=3)
        axes[row, 1].vlines(lags, 0, pacf_values, color=COLORS["orange"], linewidth=3)
        axes[row, 0].set(title=f"{name} — ACF", xlabel="lag")
        axes[row, 1].set(title=f"{name} — PACF", xlabel="lag")
        for ax in axes[row]: ax.axhline(0, color="black", linewidth=.8)
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ## 5. Estimação e AIC

    Ajustaremos ARMA como `ARIMA(order=(p,0,q))`. O AIC é
    $\mathrm{AIC}=-2\log L+2k$: recompensa ajuste e penaliza parâmetros. Só pode comparar modelos
    ajustados à mesma série e não substitui diagnóstico nem teste fora da amostra.

    Aqui, $L$ é a verossimilhança maximizada e $k$ conta parâmetros livres. Diminuir AIC significa
    que o ganho de ajuste compensou a penalidade, não que o modelo “é verdadeiro”. Diferenças
    pequenas de AIC representam evidência fraca; estabilidade, resíduos e forecast têm a palavra
    final. BIC penaliza complexidade mais fortemente conforme $n$ cresce.
    """),
    code("""
    data = simulate_arma(900, ar=[0.60], ma=[0.45], seed=303)
    candidates = []
    fitted_models = {}
    for p in range(3):
        for q in range(3):
            model = ARIMA(data, order=(p, 0, q), trend="n").fit()
            name = f"ARMA({p},{q})"
            fitted_models[name] = model
            candidates.append({"modelo": name, "p": p, "q": q, "AIC": model.aic, "BIC": model.bic})
    ranking = pd.DataFrame(candidates).sort_values("AIC").reset_index(drop=True)
    ranking.round(2)
    """),
    code("""
    best_name = ranking.loc[0, "modelo"]
    best_fit = fitted_models[best_name]
    pd.DataFrame({
        "estimativa": best_fit.params,
        "erro-padrão": best_fit.bse,
    }, index=best_fit.param_names).round(4)
    """),
    md(r"""
    ### 5.1 Resíduos ainda carregam memória?

    Um modelo vencedor por AIC ainda pode ser inadequado. Aplicamos ACF residual e Ljung–Box;
    p-valores pequenos indicam estrutura remanescente.
    """),
    code("""
    residuals = np.asarray(best_fit.resid)
    residual_acf = acf(residuals, nlags=24, fft=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.vlines(np.arange(25), 0, residual_acf, color=COLORS["purple"], linewidth=3)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(title=f"ACF residual do {best_name}", xlabel="lag", ylabel="autocorrelação")
    plt.show()
    acorr_ljungbox(residuals, lags=[5, 10, 20], return_df=True, model_df=2).round(4)
    """),
    md(r"""
    ## 6. Avaliação temporal

    Separaremos as últimas 120 observações. Cada candidato é estimado somente no treino e produz
    uma trajetória multi-step. RMSE mede ponto; cobertura e largura avaliam o intervalo.
    """),
    code("""
    train, test = data[:-120], data[-120:]
    evaluations = []
    forecasts = {}
    for p, q in [(1, 0), (0, 1), (1, 1), (2, 1)]:
        fitted = ARIMA(train, order=(p, 0, q), trend="n").fit()
        pred = fitted.get_forecast(steps=len(test))
        mean = np.asarray(pred.predicted_mean)
        interval = np.asarray(pred.conf_int(alpha=.05))
        name = f"ARMA({p},{q})"
        forecasts[name] = (mean, interval)
        evaluations.append({
            "modelo": name,
            "RMSE": np.sqrt(np.mean((test - mean)**2)),
            "cobertura": np.mean((test >= interval[:, 0]) & (test <= interval[:, 1])),
            "largura": np.mean(interval[:, 1] - interval[:, 0]),
            "AIC treino": fitted.aic,
        })
    evaluation = pd.DataFrame(evaluations).sort_values("RMSE")
    evaluation.round(3)
    """),
    code("""
    winner = evaluation.iloc[0]["modelo"]
    winner_mean, winner_interval = forecasts[winner]
    t_test = np.arange(len(train), len(data))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.arange(len(train)-80, len(train)), train[-80:], color="black", label="treino")
    ax.plot(t_test, test, color=COLORS["blue"], label="teste")
    ax.plot(t_test, winner_mean, color=COLORS["orange"], label=f"forecast {winner}")
    ax.fill_between(t_test, winner_interval[:,0], winner_interval[:,1], color=COLORS["orange"], alpha=.18)
    ax.legend(); ax.set(title="Modelo avaliado em dados não vistos", xlabel="tempo")
    plt.show()
    """),
    md(r"""
    ## 7. Exercícios

    1. Explique por que a ACF de um MA(1) corta no lag 1.
    2. Simule AR(2) com `[0.5,-0.3]` e descreva ACF/PACF.
    3. Por que escolher sempre o menor AIC pode falhar?
    4. Compare ARMA(1,1) e AR(2) em outro seed usando o mesmo split.
    5. O que um Ljung–Box pequeno pede que você faça — e o que ele não revela?
    """),
    code("""
    exercise = simulate_arma(3_000, ar=[0.5, -0.3], seed=399)
    exercise_acf = acf(exercise, nlags=5)
    exercise_pacf = pacf(exercise, nlags=5, method="ywm")
    pd.DataFrame({"lag": range(6), "ACF": exercise_acf, "PACF": exercise_pacf}).round(3)
    """),
    md(r"""
    <details><summary><strong>Gabarito resumido</strong></summary>

    1. Apenas $X_t$ e $X_{t-1}$ compartilham uma inovação; para lags maiores não há choque comum.
    2. A PACF populacional corta após 2; a ACF decai, possivelmente oscilando.
    3. AIC é relativo, dentro da amostra e não verifica resíduos nem estabilidade futura.
    4. A conclusão pode mudar por amostragem; reporte protocolo e variabilidade.
    5. Ele evidencia memória residual, mas não identifica sozinho a nova especificação.
    </details>
    """),
    code("""
    assert len(exercise) == 3_000
    assert np.isfinite(ranking[["AIC", "BIC"]].to_numpy()).all()
    assert set(evaluation["modelo"]) == {"ARMA(1,0)", "ARMA(0,1)", "ARMA(1,1)", "ARMA(2,1)"}
    print("✓ Testes rápidos passaram.")
    """),
    md(r"""
    ## Síntese

    AR(p) prolonga memória de estados; MA(q) limita memória de choques; ARMA combina ambos.
    ACF/PACF sugerem candidatos, AIC organiza parcimônia, resíduos questionam adequação e o
    backtest temporal decide utilidade preditiva. O próximo notebook trata séries que não são
    estacionárias em nível e introduz ARIMA.
    """),
]

if __name__ == "__main__":
    write_notebook(PATH, CELLS, 3)

