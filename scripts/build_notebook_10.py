from pathlib import Path

from notebook_builder import code, md, setup_cell, write_notebook


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "10_markov_switching.ipynb"


CELLS = [
    md(r"""
    # Notebook 10 — Regimes e Markov switching

    **Quando o mecanismo permanece algum tempo em um regime e depois muda**

    O filtro de Kalman usa um estado contínuo gaussiano. Aqui o estado será discreto: expansão ou
    recessão, baixa ou alta volatilidade, operação normal ou degradada. A cadeia de Markov descreve
    a persistência e o algoritmo de Hamilton atualiza probabilidades de regime a cada observação.
    """),
    md(r"""
    ## Pré-requisitos, objetivos e roteiro

    **Pré-requisitos:** probabilidade condicional, misturas gaussianas, máxima verossimilhança,
    filtragem versus suavização e Notebook 09.

    Ao final, você saberá:

    1. construir e interpretar uma matriz de transição;
    2. calcular distribuição estacionária e duração esperada de regimes;
    3. simular observações condicionadas a estados ocultos;
    4. derivar o filtro de Hamilton e o smoother de Kim;
    5. estimar parâmetros pela verossimilhança preditiva;
    6. produzir forecasts como misturas, sem escolher precocemente um único regime;
    7. distinguir mudança de regime de quebra estrutural permanente.
    """),
    md(r"""
    ## 1. Formulação matemática

    Seja $S_t\in\{0,\ldots,K-1\}$ um estado não observado. A hipótese de Markov de primeira ordem é

    $$P(S_t=j\mid S_{t-1}=i,S_{t-2},\ldots)=p_{ij}.$$

    Reunimos essas probabilidades na matriz estocástica por linha

    $$P=\begin{bmatrix}p_{00}&p_{01}\\p_{10}&p_{11}\end{bmatrix},
    \qquad p_{ij}\ge0,\quad\sum_jp_{ij}=1.$$

    No modelo gaussiano de mudança de média e variância,

    $$Y_t\mid(S_t=k)\sim N(\mu_k,\sigma_k^2).$$

    O estado é persistente, mas aleatório. Isso difere de uma dummy de período conhecida: não
    observamos diretamente a data nem o regime, apenas atualizamos suas probabilidades.
    """),
    setup_cell("from dataclasses import dataclass\nfrom scipy.optimize import minimize\nfrom scipy.special import expit, logit\nimport ipywidgets as widgets\nfrom statsmodels.tsa.regime_switching.markov_regression import MarkovRegression"),
    md(r"""
    ## 2. Persistência, distribuição estacionária e duração

    Uma distribuição estacionária de regimes satisfaz

    $$\pi=\pi P,\qquad \sum_k\pi_k=1.$$

    Se permanecemos no regime $i$ com probabilidade $p_{ii}$, sua duração $D_i$ tem distribuição
    geométrica:

    $$P(D_i=d)=p_{ii}^{d-1}(1-p_{ii}),\qquad E[D_i]=\frac{1}{1-p_{ii}}.$$

    Persistência não significa permanência. Com $p_{ii}=0{,}95$, a duração esperada é 20 períodos,
    mas sequências muito menores ou maiores continuam possíveis.
    """),
    code("""
    def stationary_distribution(transition):
        P=np.asarray(transition,dtype=float);k=P.shape[0]
        system=np.vstack([P.T-np.eye(k),np.ones(k)])
        target=np.r_[np.zeros(k),1.0]
        pi=np.linalg.lstsq(system,target,rcond=None)[0]
        return pi/pi.sum()

    P_TRUE=np.array([[.96,.04],[.10,.90]])
    MU_TRUE=np.array([-1.0,2.5])
    SIGMA_TRUE=np.array([.7,1.1])
    pi_true=stationary_distribution(P_TRUE)
    pd.DataFrame({"regime":[0,1],"probabilidade estacionária":pi_true,
        "duração esperada":1/(1-np.diag(P_TRUE))}).round(3)
    """),
    md(r"""
    ## 3. Simulação explícita

    Primeiro sorteamos $S_0\sim\pi$. Depois, a linha $S_{t-1}$ de $P$ determina a distribuição de
    $S_t$. Finalmente sorteamos $Y_t$ da emissão correspondente. A cadeia introduz blocos; a
    sobreposição das normais impede recuperar todos os estados sem incerteza.
    """),
    code("""
    def simulate_markov_switching(n,*,transition,means,stds,seed=None):
        rng=np.random.default_rng(seed);P=np.asarray(transition);means=np.asarray(means);stds=np.asarray(stds)
        states=np.empty(n,dtype=int);states[0]=rng.choice(len(means),p=stationary_distribution(P))
        for t in range(1,n):states[t]=rng.choice(len(means),p=P[states[t-1]])
        observations=rng.normal(means[states],stds[states])
        return states,observations

    states,observations=simulate_markov_switching(700,transition=P_TRUE,
        means=MU_TRUE,stds=SIGMA_TRUE,seed=1001)
    fig,ax=plt.subplots(figsize=(13,5))
    ax.plot(observations,color="black",lw=.8)
    ax.fill_between(np.arange(len(states)),observations.min()-1,observations.max()+1,
        where=states==1,color=COLORS["orange"],alpha=.16,label="regime 1 verdadeiro")
    ax.set(title="Observações produzidas por dois regimes ocultos",xlabel="tempo");ax.legend();plt.show()
    """),
    md(r"""
    ## 4. O filtro de Hamilton: previsão e atualização

    Defina $\xi_{t\mid t-1}(j)=P(S_t=j\mid y_{1:t-1})$. A etapa de previsão propaga as
    probabilidades filtradas anteriores:

    $$\xi_{t\mid t-1}(j)=\sum_iP(S_{t-1}=i\mid y_{1:t-1})p_{ij}.$$

    Após observar $y_t$, Bayes fornece

    $$\xi_{t\mid t}(j)=
    \frac{f_j(y_t)\xi_{t\mid t-1}(j)}
    {\sum_{\ell}f_{\ell}(y_t)\xi_{t\mid t-1}(\ell)},$$

    onde $f_j$ é a densidade $N(\mu_j,\sigma_j^2)$. O denominador é precisamente
    $p(y_t\mid y_{1:t-1})$, a densidade preditiva usada na verossimilhança.
    """),
    code("""
    @dataclass(frozen=True)
    class HamiltonResult:
        predicted: np.ndarray
        filtered: np.ndarray
        loglikelihood: float

    def hamilton_filter(y,*,transition,means,stds,initial=None):
        y=np.asarray(y);P=np.asarray(transition);means=np.asarray(means);stds=np.asarray(stds)
        initial=stationary_distribution(P) if initial is None else np.asarray(initial)
        predicted=np.empty((len(y),len(means)));filtered=np.empty_like(predicted)
        previous=initial;loglik=0.0
        for t,value in enumerate(y):
            predicted[t]=previous if t==0 else previous@P
            emission=stats.norm.pdf(value,loc=means,scale=stds)
            joint=predicted[t]*emission
            predictive_density=joint.sum()
            filtered[t]=joint/predictive_density
            loglik+=np.log(predictive_density)
            previous=filtered[t]
        return HamiltonResult(predicted,filtered,float(loglik))

    known_filter=hamilton_filter(observations,transition=P_TRUE,means=MU_TRUE,stds=SIGMA_TRUE)
    decoded_filtered=known_filter.filtered.argmax(axis=1)
    pd.Series({"log-verossimilhança":known_filter.loglikelihood,
        "acurácia de classificação filtrada":np.mean(decoded_filtered==states)}).round(4)
    """),
    code("""
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True)
    axes[0].plot(observations,color="black",lw=.8);axes[0].set_title("Série observada")
    axes[1].plot(known_filter.filtered[:,1],color=COLORS["blue"],label=r"$P(S_t=1|y_{1:t})$")
    axes[1].fill_between(np.arange(len(states)),0,1,where=states==1,
        color=COLORS["orange"],alpha=.16,label="regime verdadeiro")
    axes[1].axhline(.5,color="black",ls="--",lw=1);axes[1].set(ylim=(0,1),xlabel="tempo")
    axes[1].legend();plt.tight_layout();plt.show()
    """),
    md(r"""
    **Leitura:** a probabilidade não deve ser forçada prematuramente a 0 ou 1. Próximo às
    transições ou em observações compatíveis com ambos os regimes, a incerteza é parte correta da
    inferência. A classificação pelo maior valor é apenas um resumo posterior.
    """),
    md(r"""
    ## 5. Smoother de Kim

    A probabilidade filtrada usa dados até $t$. Para reconstrução histórica, podemos calcular
    $P(S_t=i\mid y_{1:T})$ com uma passagem para trás:

    $$\xi_{t\mid T}(i)=\xi_{t\mid t}(i)
    \sum_jp_{ij}\frac{\xi_{t+1\mid T}(j)}{\xi_{t+1\mid t}(j)}.$$

    Assim como no Notebook 09, probabilidades suavizadas usam o futuro e não podem ser apresentadas
    como alertas disponíveis em tempo real.
    """),
    code("""
    def kim_smoother(result,transition):
        P=np.asarray(transition);smoothed=result.filtered.copy()
        for t in range(len(smoothed)-2,-1,-1):
            ratio=smoothed[t+1]/result.predicted[t+1]
            smoothed[t]=result.filtered[t]*(P@ratio)
            smoothed[t]/=smoothed[t].sum()
        return smoothed

    known_smoothed=kim_smoother(known_filter,P_TRUE)
    pd.Series({"acurácia filtrada":np.mean(known_filter.filtered.argmax(1)==states),
        "acurácia suavizada":np.mean(known_smoothed.argmax(1)==states)}).round(4)
    """),
    md(r"""
    ## 6. Verossimilhança e estimação

    O algoritmo forward fornece

    $$\ell(\theta)=\sum_{t=1}^{T}\log p(y_t\mid y_{1:t-1};\theta).$$

    Para dois regimes, parametrizamos $p_{00}$ e $p_{11}$ por logits e os desvios por logaritmos,
    garantindo as restrições. Impomos $\mu_1>\mu_0$ escrevendo
    $\mu_1=\mu_0+\exp(\delta)$; isso resolve a troca arbitrária de rótulos durante a otimização.
    Sem uma convenção de identificação, “regime 0” e “regime 1” podem trocar nomes sem alterar a
    distribuição observável.
    """),
    code("""
    def unpack_parameters(raw):
        mu0=raw[0];mu1=mu0+np.exp(raw[1]);stds=np.exp(raw[2:4])
        p00,p11=expit(raw[4]),expit(raw[5])
        transition=np.array([[p00,1-p00],[1-p11,p11]])
        return transition,np.array([mu0,mu1]),stds

    def fit_two_regime_gaussian(y):
        q25,q75=np.quantile(y,[.25,.75])
        initial=np.array([q25,np.log(max(q75-q25,.2)),np.log(np.std(y)/2),
            np.log(np.std(y)/2),logit(.9),logit(.9)])
        def objective(raw):
            P,means,stds=unpack_parameters(raw)
            return -hamilton_filter(y,transition=P,means=means,stds=stds).loglikelihood
        result=minimize(objective,initial,method="L-BFGS-B",options={"maxiter":500})
        P,means,stds=unpack_parameters(result.x)
        return P,means,stds,hamilton_filter(y,transition=P,means=means,stds=stds),result

    P_HAT,MU_HAT,SIGMA_HAT,estimated_filter,optimization=fit_two_regime_gaussian(observations)
    parameter_table=pd.DataFrame({
        "parâmetro":["mu0","mu1","sigma0","sigma1","p00","p11"],
        "verdadeiro":[*MU_TRUE,*SIGMA_TRUE,P_TRUE[0,0],P_TRUE[1,1]],
        "estimado":[*MU_HAT,*SIGMA_HAT,P_HAT[0,0],P_HAT[1,1]]})
    parameter_table.round(3)
    """),
    md(r"""
    ## 7. Validação com implementação de referência

    `statsmodels` estima o mesmo tipo de regressão com intercepto e variância dependentes do regime.
    Diferenças pequenas podem surgir por inicialização, parametrização e tolerância numérica. Não
    usamos a biblioteca como caixa-preta: ela serve como verificação independente depois que a
    recursão e a verossimilhança já foram construídas.
    """),
    code("""
    reference=MarkovRegression(observations,k_regimes=2,trend="c",switching_variance=True).fit(
        disp=False,maxiter=300)
    reference_smoothed=np.asarray(reference.smoothed_marginal_probabilities)
    estimated_smoothed=kim_smoother(estimated_filter,P_HAT)
    # Alinhamento defensivo caso a biblioteca use a rotulagem oposta.
    corr_direct=np.corrcoef(reference_smoothed[:,1],estimated_smoothed[:,1])[0,1]
    corr_reverse=np.corrcoef(reference_smoothed[:,0],estimated_smoothed[:,1])[0,1]
    reference_regime1=reference_smoothed[:,1] if corr_direct>=corr_reverse else reference_smoothed[:,0]
    pd.Series({"loglik manual":estimated_filter.loglikelihood,"loglik statsmodels":reference.llf,
        "correlação das probabilidades suavizadas":np.corrcoef(reference_regime1,estimated_smoothed[:,1])[0,1]}).round(4)
    """),
    md(r"""
    ## 8. Forecast de regimes

    Sem novas observações, propagamos a última probabilidade filtrada:

    $$\xi_{T+h\mid T}=\xi_{T\mid T}P^h.$$

    A distribuição de $Y_{T+h}$ é uma mistura, não uma normal única:

    $$p(y_{T+h}\mid\mathcal F_T)=\sum_k\xi_{T+h\mid T}(k)
    \varphi(y;\mu_k,\sigma_k^2).$$

    Seus dois primeiros momentos são

    $$m_h=\sum_k\pi_{h,k}\mu_k,$$
    $$V_h=\sum_k\pi_{h,k}(\sigma_k^2+\mu_k^2)-m_h^2.$$

    O segundo termo inclui incerteza **entre** regimes. Substituir a mistura por apenas o regime mais
    provável subestima risco quando as probabilidades estão divididas.
    """),
    code("""
    def forecast_regime_probabilities(last_probability,transition,horizon):
        current=np.asarray(last_probability);forecasts=[]
        for _ in range(horizon):
            current=current@transition;forecasts.append(current.copy())
        return np.asarray(forecasts)

    def mixture_moments(probabilities,means,stds):
        mean=probabilities@means
        variance=probabilities@(stds**2+means**2)-mean**2
        return mean,variance

    regime_forecast=forecast_regime_probabilities(estimated_filter.filtered[-1],P_HAT,30)
    mixture_mean,mixture_variance=mixture_moments(regime_forecast,MU_HAT,SIGMA_HAT)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    axes[0].plot(np.arange(1,31),regime_forecast[:,1],marker="o",color=COLORS["blue"])
    axes[0].axhline(stationary_distribution(P_HAT)[1],color="black",ls="--",label="estacionária")
    axes[0].set(title="Probabilidade futura do regime 1",xlabel="h",ylim=(0,1));axes[0].legend()
    axes[1].plot(np.arange(1,31),mixture_mean,color=COLORS["orange"],label="média")
    axes[1].fill_between(np.arange(1,31),mixture_mean-1.96*np.sqrt(mixture_variance),
        mixture_mean+1.96*np.sqrt(mixture_variance),color=COLORS["orange"],alpha=.18)
    axes[1].set(title="Resumo normal da mistura (apenas momentos)",xlabel="h");axes[1].legend()
    plt.tight_layout();plt.show()
    """),
    md(r"""
    ### 8.1 Probabilidade de exceder um limite

    Para um limite $L$, a mistura fornece diretamente

    $$P(Y_{T+h}>L\mid\mathcal F_T)=
    \sum_k\pi_{h,k}\left[1-\Phi\left(\frac{L-\mu_k}{\sigma_k}\right)\right].$$

    Essa probabilidade preserva assimetria e multimodalidade da mistura; calcular apenas
    $1-\Phi((L-m_h)/\sqrt{V_h})$ é uma aproximação normal que pode esconder dois picos.
    """),
    code("""
    threshold=2.0
    exceedance=regime_forecast@(1-stats.norm.cdf((threshold-MU_HAT)/SIGMA_HAT))
    pd.DataFrame({"h":np.arange(1,11),"P(Y>L)":exceedance[:10]}).round(4)
    """),
    md(r"""
    ## 9. Experimento interativo: persistência

    Mantemos as emissões fixas e alteramos $p_{00}$ e $p_{11}$. Observe duração esperada,
    distribuição estacionária e a velocidade com que uma probabilidade inicial extrema perde
    memória.
    """),
    code("""
    def explore_persistence(p00=.95,p11=.90):
        P=np.array([[p00,1-p00],[1-p11,p11]])
        probabilities=forecast_regime_probabilities([1,0],P,50)
        pi=stationary_distribution(P)
        fig,ax=plt.subplots(figsize=(9,4));ax.plot(np.arange(1,51),probabilities[:,0],color=COLORS["blue"])
        ax.axhline(pi[0],color="black",ls="--");ax.set(ylim=(0,1),xlabel="h",
            title=f"E[D0]={1/(1-p00):.1f}; E[D1]={1/(1-p11):.1f}; pi0={pi[0]:.2f}")
        plt.show()
    widgets.interact(explore_persistence,
        p00=widgets.FloatSlider(min=.5,max=.99,step=.01,value=.95),
        p11=widgets.FloatSlider(min=.5,max=.99,step=.01,value=.90));
    """),
    md(r"""
    ## 10. Regime recorrente não é change point

    Markov switching presume um conjunto de regimes que pode reaparecer e uma transição
    probabilística recorrente. Change-point detection normalmente pergunta onde parâmetros mudaram
    em uma segmentação histórica, sem exigir retorno ao regime anterior.

    | Pergunta | Markov switching | Change point |
    |---|---|---|
    | estado pode voltar? | sim, governado por $P$ | não necessariamente |
    | número de mudanças | processo recorrente | inferido por segmentação/penalização |
    | saída online | probabilidade de cada regime | evidência de nova quebra |
    | incerteza principal | estado oculto e parâmetros | posição e quantidade de quebras |

    Um salto isolado também não prova mudança de regime: pode ser apenas uma observação improvável
    dentro do regime atual.
    """),
    md(r"""
    ## 11. Exercícios

    1. Calcule as durações esperadas para $p_{00}=0{,}98$ e $p_{11}=0{,}80$.
    2. Derive a distribuição estacionária de uma cadeia de dois estados.
    3. Explique por que o denominador da atualização é uma densidade preditiva.
    4. Compare probabilidade filtrada e suavizada perto de uma transição verdadeira.
    5. Faça as médias dos regimes mais próximas e meça a queda na classificação.
    6. Mostre numericamente que $\xi_{T\mid T}P^h$ converge para $\pi$.
    7. Compare a probabilidade de excedência da mistura com sua aproximação normal por momentos.
    8. Explique por que usar probabilidades suavizadas em alertas online causa leakage.
    """),
    md(r"""
    <details><summary><strong>Gabarito orientativo</strong></summary>

    1. $E[D_0]=50$ e $E[D_1]=5$.
    2. Para dois estados, $\pi_0=p_{10}/(p_{01}+p_{10})$ e $\pi_1=1-\pi_0$.
    3. Somamos a densidade conjunta sobre todos os estados possíveis.
    4. A suavizada costuma localizar melhor a transição porque usa observações posteriores.
    5. A sobreposição das emissões aumenta a incerteza sobre $S_t$.
    6. Em cadeia ergódica, a memória da distribuição inicial desaparece.
    7. A normal por momentos pode errar caudas quando a mistura é bimodal.
    8. $P(S_t\mid y_{1:T})$ contém $y_{t+1:T}$, indisponível no instante do alerta.

    </details>
    """),
    code("""
    assert optimization.success
    np.testing.assert_allclose(pi_true@P_TRUE,pi_true)
    np.testing.assert_allclose(known_filter.filtered.sum(axis=1),1)
    np.testing.assert_allclose(known_smoothed.sum(axis=1),1)
    np.testing.assert_allclose(known_smoothed[-1],known_filter.filtered[-1])
    assert np.mean(known_filter.filtered.argmax(1)==states)>.85
    np.testing.assert_allclose(regime_forecast.sum(axis=1),1)
    assert np.all((exceedance>=0)&(exceedance<=1))
    assert np.corrcoef(reference_regime1,estimated_smoothed[:,1])[0,1]>.95
    print("Testes rápidos do Notebook 10 passaram.")
    """),
    md(r"""
    ## Síntese

    Markov switching representa mecanismos recorrentes por um estado discreto persistente. O filtro
    de Hamilton combina previsão pela cadeia e atualização por Bayes; o smoother revisa o passado;
    a verossimilhança resulta das densidades preditivas; e o forecast continua sendo uma mistura.

    **Armadilhas:** interpretar rótulos sem convenção; usar estado mais provável e descartar
    incerteza; confundir suavização com inferência online; criar regimes demais; e chamar toda
    observação extrema de transição.

    O Notebook 11 mudará a pergunta: em vez de regimes recorrentes conhecidos, estimaremos quantas
    quebras estruturais ocorreram e em quais posições, controlando o custo de adicionar segmentos.
    """),
]


if __name__ == "__main__":
    write_notebook(PATH, CELLS, 10)
