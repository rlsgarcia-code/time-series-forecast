from pathlib import Path

from notebook_builder import code, md, setup_cell, write_notebook

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "06_arch_garch.ipynb"

CELLS = [
    md(r"""
    # Notebook 06 — Volatilidade ARCH/GARCH

    **Quando a média é pouco previsível, mas a intensidade do risco tem memória**

    Retornos financeiros frequentemente têm ACF pequena em nível e forte dependência nos
    quadrados. Construiremos ARCH/GARCH, estimaremos por máxima verossimilhança e produziremos
    intervalos cuja largura muda no tempo.
    """),
    md(r"""
    ## Objetivos e roteiro

    Reconhecer clustering de volatilidade; separar retorno $r_t$ de variância condicional $h_t$;
    simular e estimar GARCH(1,1); interpretar persistência e variância de longo prazo; diagnosticar
    resíduos padronizados; prever volatilidade e avaliar intervalos/VaR.
    """),
    md(r"""
    ## Mapa conceitual: variância condicional não é variância amostral móvel

    A informação disponível antes de $t$ é $\mathcal F_{t-1}$. GARCH modela
    $h_t=\mathrm{Var}(r_t\mid\mathcal F_{t-1})$: uma previsão feita **antes** de observar $r_t$.
    Já a variância incondicional descreve a dispersão de longo prazo depois de misturar períodos
    calmos e turbulentos.

    | Quantidade | Quando é conhecida? | Papel |
    |---|---|---|
    | $z_t$ | nunca antecipadamente | choque padronizado, média 0 e variância 1 |
    | $h_t$ | em $t-1$, dados parâmetros e histórico | escala prevista para $r_t$ |
    | $r_t=\sqrt{h_t}z_t$ | observado em $t$ | retorno realizado |
    | $E[h_t]$ | propriedade de longo prazo | nível ao qual forecasts convergem |

    Clustering não significa que retornos terão o mesmo sinal; significa que grandes magnitudes
    tendem a ser seguidas por grandes magnitudes.
    """),
    md(r"""
    ## Espinha matemática: distribuição condicional GARCH(1,1)

    O modelo completo especifica retorno e variância:

    $$r_t=\mu+\sqrt{h_t}z_t,\qquad z_t\overset{iid}{\sim}N(0,1),$$
    $$h_t=\omega+\alpha(r_{t-1}-\mu)^2+\beta h_{t-1}.$$

    Com $\omega>0$, $\alpha,\beta\ge0$ e $\alpha+\beta<1$, a variância incondicional existe. Tomando
    esperança na recorrência e usando $E[(r_{t-1}-\mu)^2]=E[h_{t-1}]$,

    $$\bar h=E[h_t]=\omega+(\alpha+\beta)\bar h
    \quad\Longrightarrow\quad
    \bar h=\frac{\omega}{1-\alpha-\beta}.$$

    Sob normalidade condicional, a contribuição de $t$ à log-verossimilhança negativa, ignorando
    uma constante, é

    $$\ell_t(\theta)=\tfrac12\left[\log h_t(\theta)
    +\frac{(r_t-\mu)^2}{h_t(\theta)}\right].$$

    A previsão fechada de variância é

    $$E_T[h_{T+h}]=\bar h+(\alpha+\beta)^{h-1}
    \bigl(h_{T+1}-\bar h\bigr),\qquad h\ge1.$$

    Assim, $\alpha+\beta$ controla a velocidade de retorno ao risco de longo prazo. Para a
    convenção de retorno, o quantil inferior é
    $q_p(r_{T+1}\mid\mathcal F_T)=\mu+\sqrt{h_{T+1}}\Phi^{-1}(p)$; ao reportar VaR como perda
    positiva, o sinal precisa ser invertido explicitamente.
    """),
    setup_cell("from scipy.optimize import minimize\nfrom statsmodels.tsa.stattools import acf\nfrom statsmodels.stats.diagnostic import acorr_ljungbox"),
    md(r"""
    ## 1. Do ruído homocedástico ao ARCH

    Em ruído branco, $\mathrm{Var}(r_t\mid\mathcal F_{t-1})=\sigma^2$ é constante. Em ARCH(1):
    $r_t=\sqrt{h_t}z_t$, $h_t=\omega+\alpha r_{t-1}^2$, $z_t\sim N(0,1)$.
    Um retorno extremo aumenta a variância do próximo período independentemente do sinal.

    O quadrado aparece porque risco mede magnitude e porque
    $E[r_t^2\mid\mathcal F_{t-1}]=h_t$ quando $E[z_t^2]=1$. ARCH puro lembra choques observados;
    GARCH adiciona a própria previsão anterior de variância, produzindo memória mais longa com
    poucos parâmetros.
    """),
    code("""
    def simulate_garch11(n,*,omega,alpha,beta,seed=None):
        '''Simula r_t=sqrt(h_t)z_t e h_t=omega+alpha r²_(t-1)+beta h_(t-1).'''
        if omega<=0 or min(alpha,beta)<0 or alpha+beta>=1: raise ValueError("GARCH não estacionário")
        rng=np.random.default_rng(seed); z=rng.normal(size=n)
        r=np.empty(n); h=np.empty(n); h[0]=omega/(1-alpha-beta); r[0]=np.sqrt(h[0])*z[0]
        for t in range(1,n):
            h[t]=omega+alpha*r[t-1]**2+beta*h[t-1]
            r[t]=np.sqrt(h[t])*z[t]
        return r,h

    TRUE={"omega":.04,"alpha":.08,"beta":.88}
    returns,variance=simulate_garch11(3_000,**TRUE,seed=601)
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    axes[0].plot(returns,color=COLORS["blue"],lw=.8); axes[0].set_title("Retornos: clusters de amplitude")
    axes[1].plot(np.sqrt(variance),color=COLORS["orange"]); axes[1].set_title("Volatilidade condicional latente")
    axes[1].set_xlabel("tempo");plt.tight_layout();plt.show()
    """),
    md(r"""
    ## 2. GARCH(1,1): choque e persistência

    $$h_t=\omega+\alpha r_{t-1}^2+\beta h_{t-1}.$$

    $\alpha$ mede reação a notícias; $\beta$ persistência da volatilidade; $\alpha+\beta$ controla
    a memória total. Para $\alpha+\beta<1$, $E[h_t]=\omega/(1-\alpha-\beta)$.

    Interprete os parâmetros em conjunto: `alpha` alto produz saltos rápidos após choques; `beta`
    alto produz decaimento lento. A soma próxima de 1 implica meia-vida longa e grande sensibilidade
    da variância incondicional. A restrição $\alpha+\beta<1$ é condição de covariância-estacionariedade
    sob o modelo, não garantia de que dados reais nunca mudem de regime.
    """),
    code("""
    long_run_variance=TRUE["omega"]/(1-TRUE["alpha"]-TRUE["beta"])
    half_life=np.log(.5)/np.log(TRUE["alpha"]+TRUE["beta"])
    pd.Series({"alpha+beta":TRUE["alpha"]+TRUE["beta"],"variância de longo prazo":long_run_variance,
        "volatilidade de longo prazo":np.sqrt(long_run_variance),"meia-vida do choque":half_life}).round(3)
    """),
    md(r"""
    ### 2.1 ACF do retorno versus ACF do quadrado

    GARCH não precisa tornar $r_t$ linearmente previsível; torna $r_t^2$ previsível. Por isso
    inspecionamos retornos absolutos/quadrados, não apenas ACF em nível.
    """),
    code("""
    lags=np.arange(31)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].vlines(lags,0,acf(returns,nlags=30),color=COLORS["blue"],lw=3); axes[0].set_title("ACF dos retornos")
    axes[1].vlines(lags,0,acf(returns**2,nlags=30),color=COLORS["red"],lw=3); axes[1].set_title("ACF dos retornos²")
    for ax in axes: ax.axhline(0,color="black",lw=.8);ax.set_xlabel("lag")
    plt.tight_layout();plt.show()
    """),
    md(r"""
    ## 3. Filtragem e máxima verossimilhança

    Dado $(\omega,\alpha,\beta)$ e os retornos, calculamos $h_t$ recursivamente. Sob normalidade,
    ignorando constante: $-\log L=\frac12\sum_t[\log h_t+r_t^2/h_t]$. Minimizamos esse valor com
    as restrições de positividade e estacionariedade.

    A função objetivo equilibra dois termos: `log(h_t)` penaliza variâncias sempre grandes;
    `r_t²/h_t` penaliza variâncias pequenas diante de retornos extremos. Para cada candidato de
    parâmetros, toda a sequência $h_1,\ldots,h_T$ precisa ser recalculada. O valor inicial de
    $h_0$ é uma aproximação; sua influência diminui com uma série suficientemente longa.

    Máxima verossimilhança gaussiana estima uma *quasi-likelihood* útil mesmo sem normalidade exata,
    mas erros-padrão e quantis de cauda exigem cuidados adicionais.
    """),
    code("""
    def garch_variance(r,omega,alpha,beta):
        r=np.asarray(r);h=np.empty(len(r));h[0]=omega/(1-alpha-beta)
        for t in range(1,len(r)): h[t]=omega+alpha*r[t-1]**2+beta*h[t-1]
        return h

    def fit_garch11(r):
        r=np.asarray(r)
        def nll(params):
            omega,alpha,beta=params
            if omega<=0 or min(alpha,beta)<0 or alpha+beta>=.999:return 1e100
            h=garch_variance(r,omega,alpha,beta)
            return .5*np.sum(np.log(h)+r**2/h)
        v=np.var(r)
        result=minimize(nll,[.05*v,.08,.88],method="SLSQP",bounds=[(1e-9,None),(0,.999),(0,.999)],
            constraints=[{"type":"ineq","fun":lambda p:.999-p[1]-p[2]}],options={"maxiter":500,"ftol":1e-10})
        params=result.x; return params,garch_variance(r,*params),result

    estimates,filtered,optimization=fit_garch11(returns)
    pd.DataFrame({"parâmetro":["omega","alpha","beta"],"verdadeiro":list(TRUE.values()),"estimado":estimates}).round(4)
    """),
    md(r"""
    ## 4. Diagnóstico com resíduos padronizados

    Se o modelo capturou a variância condicional, $z_t=r_t/\sqrt{\widehat h_t}$ deve ter variância
    próxima de 1 e pouca autocorrelação tanto em $z_t$ quanto em $z_t^2$. Caudas pesadas podem
    permanecer, sugerindo inovações Student-t. O teste de **Ljung–Box** resume a evidência de
    autocorrelação em vários lags para as duas sequências.

    Há dois diagnósticos diferentes: ACF de $z_t$ pergunta se sobrou previsibilidade na média;
    ACF de $z_t^2$ pergunta se sobrou previsibilidade na variância. Um modelo pode passar no
    primeiro e falhar no segundo. O histograma/Q–Q avalia uma terceira hipótese: o formato da
    distribuição condicional.
    """),
    code("""
    standardized=returns/np.sqrt(filtered)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].hist(standardized,bins=50,density=True,color=COLORS["blue"],alpha=.7)
    grid=np.linspace(-4,4,300);axes[0].plot(grid,stats.norm.pdf(grid),color=COLORS["orange"]);axes[0].set_title("Resíduos padronizados")
    axes[1].vlines(lags,0,acf(standardized,nlags=30),color=COLORS["green"],lw=3);axes[1].set_title("ACF de z")
    axes[2].vlines(lags,0,acf(standardized**2,nlags=30),color=COLORS["purple"],lw=3);axes[2].set_title("ACF de z²")
    for ax in axes[1:]:ax.axhline(0,color="black",lw=.8);ax.set_xlabel("lag")
    plt.tight_layout();plt.show()
    pd.DataFrame({"z":acorr_ljungbox(standardized,lags=[10,20],return_df=True)["lb_pvalue"],
        "z²":acorr_ljungbox(standardized**2,lags=[10,20],return_df=True)["lb_pvalue"]}).round(4)
    """),
    md(r"""
    ## 5. Forecast de variância

    Um passo: $h_{T+1}=\omega+\alpha r_T^2+\beta h_T$. Para horizontes seguintes, substituímos
    $E_T[r_{T+h}^2]=h_{T+h}$, então
    $h_{T+h}=\omega+(\alpha+\beta)h_{T+h-1}$. A previsão converge à variância de longo prazo.
    """),
    code("""
    def forecast_garch_variance(last_return,last_variance,horizon,omega,alpha,beta):
        forecasts=np.empty(horizon)
        forecasts[0]=omega+alpha*last_return**2+beta*last_variance
        for h in range(1,horizon):forecasts[h]=omega+(alpha+beta)*forecasts[h-1]
        return forecasts

    h_forecast=forecast_garch_variance(returns[-1],filtered[-1],60,*estimates)
    fit_long_run=estimates[0]/(1-estimates[1]-estimates[2])
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(np.sqrt(h_forecast),color=COLORS["orange"],lw=2)
    ax.axhline(np.sqrt(fit_long_run),color=COLORS["green"],ls="--",label="longo prazo")
    ax.set(title="Forecast da volatilidade",xlabel="horizonte",ylabel="desvio-padrão");ax.legend();plt.show()
    """),
    md(r"""
    ## 6. Intervalos dinâmicos e VaR

    Com média zero e normalidade, um intervalo de 95% é $\pm1{,}96\sqrt{h_t}$. O VaR inferior de
    1% é $-2{,}326\sqrt{h_t}$. Avaliamos exceções fora da amostra; cobertura correta não garante
    independência das exceções.

    VaR de 1% não é “a pior perda possível”. É um quantil: sob calibração, cerca de 1% dos retornos
    devem ficar abaixo dele. Exceções agrupadas indicam que o modelo reage tarde ao risco, mesmo se
    a taxa total estiver próxima de 1%. Para medir severidade além do quantil, usaríamos Expected
    Shortfall.
    """),
    code("""
    train,test=returns[:2400],returns[2400:]
    params,h_train,_=fit_garch11(train)
    omega,alpha,beta=params; current_h=h_train[-1]; current_r=train[-1]
    variances=[]
    for realized in test:
        next_h=omega+alpha*current_r**2+beta*current_h
        variances.append(next_h);current_h=next_h;current_r=realized
    variances=np.asarray(variances);std=np.sqrt(variances)
    lower,upper=stats.norm.ppf(.025)*std,stats.norm.ppf(.975)*std
    var01=stats.norm.ppf(.01)*std
    pd.Series({"cobertura 95%":np.mean((test>=lower)&(test<=upper)),
        "taxa de exceção VaR 1%":np.mean(test<var01),"largura média":np.mean(upper-lower)}).round(4)
    """),
    code("""
    fig,ax=plt.subplots(figsize=(12,5));t=np.arange(len(test))
    ax.plot(t,test,color=COLORS["blue"],lw=.8,label="retorno")
    ax.plot(t,upper,color=COLORS["orange"],lw=1,label="intervalo 95%")
    ax.plot(t,lower,color=COLORS["orange"],lw=1);ax.fill_between(t,lower,upper,color=COLORS["orange"],alpha=.12)
    ax.scatter(t[test<var01],test[test<var01],color=COLORS["red"],s=20,label="exceção VaR 1%")
    ax.legend(ncol=3);ax.set(title="Intervalos adaptam largura ao risco",xlabel="tempo");plt.show()
    """),
    md(r"""
    ## 7. Exercícios

    1. Calcule variância incondicional para $(.04,.08,.88)$. 2. Compare meia-vida quando
    $\alpha+\beta=.8$ e .98. 3. Por que ACF dos retornos pode ser zero e dos quadrados não?
    4. Simule um choque grande e acompanhe $h_t$. 5. Explique por que normalidade pode subestimar VaR.
    """),
    md(r"""
    <details><summary><strong>Gabarito comentado</strong></summary>

    1. $0{,}04/(1-0{,}08-0{,}88)=1$. 2. Persistência 0,98 tem meia-vida muito maior.
    3. Sinais podem ser imprevisíveis enquanto magnitudes dependem do passado. 4. O efeito decai
    aproximadamente como $(\alpha+\beta)^h$. 5. A normal tem caudas mais leves que muitos retornos;
    Student-t pode representar melhor eventos extremos.
    </details>
    """),
    code("""
    assert optimization.success
    assert estimates[0]>0 and estimates[1]>=0 and estimates[2]>=0 and estimates[1]+estimates[2]<1
    assert abs(np.var(standardized)-1)<.08
    assert np.all(h_forecast>0)
    print("✓ Gabarito estrutural e testes rápidos passaram.")
    """),
    md(r"""
    ## Síntese

    GARCH modela a informação contida na magnitude dos choques. Persistência controla a velocidade
    de retorno ao risco de longo prazo; resíduos padronizados verificam o que sobrou. Forecasts de
    variância produzem intervalos dinâmicos, mas distribuição das inovações e backtesting de
    cobertura continuam essenciais. O próximo notebook formaliza essa avaliação probabilística.
    """),
]

if __name__ == "__main__":
    write_notebook(PATH,CELLS,6)

