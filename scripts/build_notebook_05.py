from pathlib import Path

from notebook_builder import code, md, setup_cell, write_notebook

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "05_browniano_gbm_ito.ipynb"

CELLS = [
    md(r"""
    # Notebook 05 — Movimento Browniano, GBM e Itô

    **Do passeio aleatório ao tempo contínuo**

    Construiremos o movimento Browniano a partir de incrementos gaussianos, mediremos sua
    variação quadrática, simularemos movimento Browniano geométrico e usaremos o lema de Itô para
    explicar o termo $-\sigma^2/2$ na dinâmica logarítmica.
    """),
    md(r"""
    ## Objetivos e roteiro

    Entender incrementos $\Delta W\sim N(0,\Delta t)$, escala $\sqrt{\Delta t}$, continuidade sem
    diferenciabilidade, variação quadrática, SDE, discretização Euler–Maruyama, solução exata do
    GBM e transformação por Itô. Pré-requisito: passeio aleatório e normal.
    """),
    md(r"""
    ## Mapa conceitual: do passo discreto ao limite contínuo

    Para que a variância total permaneça finita quando diminuímos o passo, cada incremento de
    duração $\Delta t$ precisa ter variância $\Delta t$ e desvio-padrão $\sqrt{\Delta t}$:

    $$n\,\mathrm{Var}(\Delta W)=\frac{T}{\Delta t}\Delta t=T.$$

    Com escala $\Delta t$, a aleatoriedade desapareceria no limite; com escala constante,
    explodiria. O Browniano é o limite não trivial entre passos menores e mais numerosos.

    | Símbolo | Leitura correta |
    |---|---|
    | $W_t$ | estado acumulado do movimento Browniano |
    | $dW_t$ | incremento aleatório de ordem $\sqrt{dt}$ |
    | $dt$ | incremento temporal determinístico |
    | SDE | regra para incrementos/integrais, não uma derivada comum |
    """),
    md(r"""
    ## Espinha matemática: Browniano, SDE e lema de Itô

    Um movimento Browniano padrão $(W_t)_{t\ge0}$ satisfaz: $W_0=0$; possui trajetórias contínuas;
    incrementos em intervalos disjuntos são independentes; e, para $0\le s<t$,

    $$W_t-W_s\sim N(0,t-s).$$

    Em particular,

    $$E[W_t]=0,\qquad \operatorname{Var}(W_t)=t,\qquad
    \operatorname{Cov}(W_s,W_t)=\min(s,t).$$

    Uma equação diferencial estocástica

    $$dX_t=a(t,X_t)dt+b(t,X_t)dW_t$$

    é abreviação da equação integral

    $$X_t=X_0+\int_0^t a(s,X_s)ds+\int_0^t b(s,X_s)dW_s.$$

    Para $f\in C^{1,2}$, o lema de Itô dá

    $$df(t,X_t)=\left(f_t+a f_x+\tfrac12b^2f_{xx}\right)dt+b f_xdW_t.$$

    O termo de segunda ordem permanece porque $(dW_t)^2$ é de ordem $dt$. No GBM,

    $$dS_t=\mu S_tdt+\sigma S_tdW_t,$$

    aplicar Itô a $f(S)=\log S$ produz

    $$S_{t+h}\mid\mathcal F_t
    =S_t\exp\!\left[(\mu-\tfrac12\sigma^2)h+\sigma\sqrt h Z\right],quad Z\sim N(0,1).$$

    Portanto, $\log(S_{t+h}/S_t)$ é normal, enquanto $S_{t+h}$ é lognormal, com
    $E_t[S_{t+h}]=S_te^{\mu h}$.
    """),
    setup_cell(),
    md(r"""
    ## 1. Movimento Browniano padrão

    $W_0=0$; incrementos em intervalos disjuntos são independentes; e
    $W_{t+\Delta t}-W_t\sim N(0,\Delta t)$. Logo, o desvio do incremento é
    $\sqrt{\Delta t}$ — não $\Delta t$.

    Incrementos independentes não significam níveis independentes: $W_t$ e $W_{t+h}$ compartilham
    todo o caminho até $t$. Os caminhos são contínuos, mas quase certamente não diferenciáveis;
    ampliar o gráfico revela novas irregularidades em qualquer escala.
    """),
    code("""
    def brownian_paths(n_steps, *, horizon=1.0, n_paths=1, seed=None):
        '''Gera W_t acumulando incrementos N(0, dt).'''
        dt = horizon / n_steps
        rng = np.random.default_rng(seed)
        dW = rng.normal(0, np.sqrt(dt), size=(n_paths,n_steps))
        W = np.column_stack([np.zeros(n_paths),np.cumsum(dW,axis=1)])
        return np.linspace(0,horizon,n_steps+1), W

    time,W100 = brownian_paths(1000,horizon=1,n_paths=100,seed=501)
    fig,ax=plt.subplots(figsize=(11,5))
    ax.plot(time,W100.T,color=COLORS["blue"],alpha=.10,lw=.8)
    ax.plot(time,W100.mean(axis=0),color="black",lw=2,label="média de 100 caminhos")
    ax.set(title="100 trajetórias Brownianas",xlabel="tempo",ylabel=r"$W_t$"); ax.legend(); plt.show()
    """),
    code("""
    _,many = brownian_paths(500,horizon=2,n_paths=20_000,seed=502)
    terminal = many[:,-1]
    pd.Series({"média empírica":terminal.mean(),"média teórica":0,
        "variância empírica":terminal.var(),"variância teórica":2}).round(4)
    """),
    md(r"""
    ## 2. A escala correta e a variação quadrática

    Se dividimos $[0,T]$ em $n$ partes, cada incremento tem variância $T/n$. A soma dos quadrados
    converge para $T$: $\sum(\Delta W)^2\to T$. Já a soma das variações absolutas explode. Essa
    variação quadrática não nula é a origem do termo de segunda derivada no cálculo de Itô.

    A contagem de ordens explica o fenômeno: $\Delta W=O(\sqrt{\Delta t})$, então
    $(\Delta W)^2=O(\Delta t)$ e sobrevive. Já
    $\Delta t\Delta W=O(\Delta t^{3/2})$ e $(\Delta t)^2$ desaparecem mais rápido. Escrevemos
    simbolicamente $(dW)^2=dt$, $dt\,dW=0$ e $(dt)^2=0$.
    """),
    code("""
    def quadratic_variation(path):
        return np.sum(np.diff(path)**2)

    resolutions=[50,200,1000,5000]
    qv=[]
    for n in resolutions:
        _,paths=brownian_paths(n,horizon=1,n_paths=1_000,seed=510+n)
        qv.append(np.mean(np.sum(np.diff(paths,axis=1)**2,axis=1)))
    pd.DataFrame({"passos":resolutions,"variação quadrática média":qv,"limite teórico":1}).round(4)
    """),
    md(r"""
    ## 3. Equações diferenciais estocásticas

    A notação $dX_t=a(X_t,t)dt+b(X_t,t)dW_t$ separa drift previsível e difusão aleatória.
    Euler–Maruyama aproxima em passos:
    $X_{t+\Delta t}=X_t+a(X_t,t)\Delta t+b(X_t,t)\Delta W_t$.

    Para GBM: $dS_t=\mu S_tdt+\sigma S_tdW_t$.

    Não dividimos por `dt` para interpretar $dW_t/dt$ como velocidade finita. Condicionado a
    $X_t$, em um passo curto, $E[\Delta X]\approx a\Delta t$ e
    $\mathrm{Var}(\Delta X)\approx b^2\Delta t$: `a` move o centro e `b` controla a dispersão local.
    """),
    code("""
    def gbm_euler(s0,mu,sigma,n_steps,*,horizon=1,n_paths=1,seed=None):
        dt=horizon/n_steps
        rng=np.random.default_rng(seed)
        dW=rng.normal(0,np.sqrt(dt),size=(n_paths,n_steps))
        S=np.empty((n_paths,n_steps+1)); S[:,0]=s0
        for t in range(n_steps):
            S[:,t+1]=S[:,t]+mu*S[:,t]*dt+sigma*S[:,t]*dW[:,t]
        return np.linspace(0,horizon,n_steps+1),S

    t,euler=gbm_euler(100,.08,.25,252,n_paths=100,seed=520)
    fig,ax=plt.subplots(figsize=(11,5)); ax.plot(t,euler.T,alpha=.10,color=COLORS["orange"])
    ax.set(title="100 trajetórias GBM por Euler–Maruyama",xlabel="tempo",ylabel=r"$S_t$"); plt.show()
    """),
    md(r"""
    ## 4. Lema de Itô e a solução exata do GBM

    Para $dX_t=a\,dt+b\,dW_t$ e $f(X_t,t)$:
    $$df=(f_t+af_x+\tfrac12b^2f_{xx})dt+bf_xdW_t.$$

    Aplicando $f(S)=\log S$: $f'=1/S$, $f''=-1/S^2$. Portanto
    $$d\log S_t=(\mu-\tfrac12\sigma^2)dt+\sigma dW_t.$$
    O termo $-\sigma^2/2$ é a correção de Itô. Integrando:
    $$S_t=S_0\exp[(\mu-\sigma^2/2)t+\sigma W_t].$$

    A fórmula nasce de Taylor até segunda ordem. No cálculo comum, termos quadráticos desaparecem;
    aqui, $\tfrac12f_{xx}(dX)^2$ contém
    $\tfrac12f_{xx}b^2(dW)^2=\tfrac12f_{xx}b^2dt$ e sobrevive. A correção de Itô não é um ajuste
    arbitrário, mas consequência da variação quadrática.

    No GBM, $\mu$ é a taxa de crescimento de $E[S_t]$, enquanto $\mu-\sigma^2/2$ é o drift de
    $\log S_t$. Essa diferença explica por que média e mediana da lognormal não coincidem.
    """),
    code("""
    def gbm_exact(s0,mu,sigma,n_steps,*,horizon=1,n_paths=1,seed=None):
        time,W=brownian_paths(n_steps,horizon=horizon,n_paths=n_paths,seed=seed)
        return time,s0*np.exp((mu-.5*sigma**2)*time+sigma*W)

    t,exact=gbm_exact(100,.08,.25,252,n_paths=20_000,seed=521)
    terminal=exact[:,-1]
    theoretical_mean=100*np.exp(.08)
    theoretical_median=100*np.exp(.08-.5*.25**2)
    pd.Series({"média empírica":terminal.mean(),"média teórica":theoretical_mean,
        "mediana empírica":np.median(terminal),"mediana teórica":theoretical_median}).round(3)
    """),
    md(r"""
    Média e mediana diferem porque $S_t$ é lognormal e assimétrico. A correção de Itô reduz o
    drift do log, mas $E[S_t]=S_0e^{\mu t}$ continua contendo $\mu$ devido à convexidade da exponencial.
    """),
    code("""
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].hist(terminal,bins=60,density=True,color=COLORS["blue"],alpha=.75)
    axes[0].set(title=r"Distribuição de $S_T$ (lognormal)",xlabel=r"$S_T$")
    log_return=np.log(terminal/100)
    axes[1].hist(log_return,bins=60,density=True,color=COLORS["purple"],alpha=.75)
    grid=np.linspace(log_return.min(),log_return.max(),300)
    axes[1].plot(grid,stats.norm.pdf(grid,.08-.5*.25**2,.25),color=COLORS["orange"])
    axes[1].set(title=r"Distribuição de $\\log(S_T/S_0)$",xlabel="log-retorno")
    plt.tight_layout();plt.show()
    """),
    md(r"""
    ## 5. Euler versus solução exata

    Usando os mesmos incrementos, a diferença é apenas erro de discretização. Euler pode até gerar
    valores negativos em passos grosseiros ou volatilidade alta; a solução exata preserva positividade.
    """),
    code("""
    def compare_euler_exact(s0,mu,sigma,n_steps,seed=0):
        dt=1/n_steps; rng=np.random.default_rng(seed); dW=rng.normal(0,np.sqrt(dt),n_steps)
        W=np.r_[0,np.cumsum(dW)]
        e=np.empty(n_steps+1); e[0]=s0
        for i in range(n_steps): e[i+1]=e[i]+mu*e[i]*dt+sigma*e[i]*dW[i]
        time=np.linspace(0,1,n_steps+1)
        exact=s0*np.exp((mu-.5*sigma**2)*time+sigma*W)
        return time,e,exact

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,n in zip(axes,[12,252]):
        time,e,x=compare_euler_exact(100,.08,.35,n,seed=530)
        ax.plot(time,x,label="exata",color=COLORS["blue"]); ax.plot(time,e,"--",label="Euler",color=COLORS["orange"])
        ax.set(title=f"{n} passos — erro final {abs(e[-1]-x[-1]):.3f}",xlabel="tempo"); ax.legend()
    plt.tight_layout();plt.show()
    """),
    md(r"""
    ## 6. Itô aplicado a $S_t^2$

    Para $f(S)=S^2$, $f'=2S$ e $f''=2$. Logo
    $d(S_t^2)=(2\mu+\sigma^2)S_t^2dt+2\sigma S_t^2dW_t$. O termo $\sigma^2S_t^2dt$ não aparece
    no cálculo comum; vem de $(dW_t)^2=dt$ em variação quadrática.
    """),
    code("""
    mu,sigma=.08,.25
    empirical_second=np.mean(terminal**2)
    theoretical_second=100**2*np.exp((2*mu+sigma**2))
    pd.Series({"E[S_T²] empírico":empirical_second,"E[S_T²] por Itô":theoretical_second,
        "erro relativo":empirical_second/theoretical_second-1}).round(4)
    """),
    md(r"""
    ## 7. Exercícios

    1. Por que o incremento usa $\sqrt{dt}$? 2. Simule $W_2$ e verifique variância 2.
    3. Derive Itô para $f(W)=W^2$. 4. Explique por que GBM permanece positivo.
    5. Compare erro de Euler para 12, 52 e 252 passos com o mesmo caminho.
    """),
    md(r"""
    <details><summary><strong>Gabarito comentado</strong></summary>

    1. Porque variâncias somam: `n` incrementos com variância `dt=T/n` resultam em variância T.
    2. $W_2\sim N(0,2)$. 3. $d(W_t^2)=dt+2W_tdW_t$. 4. A exponencial é sempre positiva.
    5. Com o mesmo caminho, o erro de discretização tende a cair quando a malha fica mais fina.
    </details>
    """),
    code("""
    _,check=brownian_paths(800,horizon=2,n_paths=15_000,seed=599)
    assert abs(check[:,-1].var()-2)<.08
    assert abs(np.mean([quadratic_variation(p) for p in check[:500]])-2)<.12
    assert np.all(exact>0)
    assert abs(terminal.mean()/theoretical_mean-1)<.02
    print("✓ Gabarito numérico e testes rápidos passaram.")
    """),
    md(r"""
    ## Síntese

    Browniano é o limite contínuo do passeio aleatório com escala $\sqrt{dt}$. Sua variação
    quadrática muda a regra da cadeia e produz a correção de Itô. GBM transforma Browniano em um
    processo positivo e lognormal. No próximo notebook, a variância deixará de ser constante e
    passará a responder a choques passados via ARCH/GARCH.
    """),
]

if __name__ == "__main__":
    write_notebook(PATH,CELLS,5)

