from pathlib import Path

from notebook_builder import code, md, setup_cell, write_notebook


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "12_cusum_ewma.ipynb"


CELLS = [
    md(r"""
    # Notebook 12 — Detecção sequencial com CUSUM e EWMA

    **Acumular evidência sem confundir repetição de testes com monitoramento calibrado**

    Change-point offline usa o futuro para segmentar o passado. Agora processaremos uma observação
    por vez e perguntaremos quando a evidência disponível naquele instante justifica um alerta.
    """),
    md(r"""
    ## Pré-requisitos, objetivos e roteiro

    **Pré-requisitos:** resíduos preditivos, normal padronizada, razão de verossimilhança,
    simulação Monte Carlo e Notebook 11.

    Ao final, você saberá:

    1. transformar forecasts preservados em surpresas prequentialmente válidas;
    2. derivar o CUSUM como acumulação de log-razões de verossimilhança;
    3. derivar média e variância transitória do EWMA;
    4. interpretar ARL, falso alarme, probabilidade de detecção e atraso;
    5. calibrar limiares por simulação sob um modelo nulo;
    6. comparar sensibilidade para mudanças de tamanhos diferentes;
    7. reconhecer falhas de calibração causadas por autocorrelação e escala errada.
    """),
    md(r"""
    ## 1. A entrada correta é a surpresa prequential

    Na origem $t-1$, o modelo emite média e desvio-padrão para $Y_t$. Depois que $Y_t$ chega,
    calculamos

    $$Z_t=\frac{Y_t-\widehat\mu_{t\mid t-1}}
    {\widehat\sigma_{t\mid t-1}}.$$

    Sob um modelo normal corretamente calibrado, esperamos aproximadamente $Z_t\sim N(0,1)$ sem
    dependência serial. O detector deve receber a previsão **congelada antes do realizado**. Usar
    resíduos recalculados após retreinar com $Y_t$ reduz artificialmente a surpresa.

    Neste laboratório simularemos diretamente $Z_t$. Assim isolamos o detector da qualidade do
    modelo de forecasting; na integração final as duas camadas serão conectadas.
    """),
    setup_cell("from dataclasses import dataclass\nimport ipywidgets as widgets"),
    md(r"""
    ## 2. Por que testar cada ponto a 99% não controla a sequência?

    Se cada $Z_t$ independente gera alerta quando $|Z_t|>2{,}576$, a probabilidade pontual é 1%.
    Em $T$ inspeções, a probabilidade de pelo menos um falso alerta é

    $$P(\text{ao menos um})=1-(1-0{,}01)^T.$$

    Para $T=500$, isso supera 99%. Monitoramento contínuo exige métricas sequenciais; um limiar
    pontual não possui o mesmo significado depois de centenas de oportunidades.
    """),
    code("""
    horizons=np.array([1,10,50,100,500,1000])
    pd.DataFrame({"inspeções":horizons,
        "P(>=1 falso alerta)":1-(1-.01)**horizons}).round(4)
    """),
    md(r"""
    ## 3. Cenário controlado

    Geraremos 180 observações sob $N(0,1)$ e depois uma mudança persistente para $N(0{,}8,1)$.
    Pontos antes de 180 são falsos alarmes; um alerta posterior é uma detecção com atraso
    $T_{alarme}-180$.
    """),
    code("""
    def simulate_shift(n_pre,n_post,*,shift,seed=None):
        rng=np.random.default_rng(seed)
        return np.r_[rng.normal(size=n_pre),rng.normal(loc=shift,size=n_post)]

    CHANGE_AT=180
    surprises=simulate_shift(CHANGE_AT,220,shift=.8,seed=1201)
    point_alert=np.abs(surprises)>stats.norm.ppf(.995)
    fig,ax=plt.subplots(figsize=(12,4));ax.plot(surprises,color=COLORS["gray"],lw=.8)
    ax.scatter(np.flatnonzero(point_alert),surprises[point_alert],color=COLORS["red"],label="alerta pontual 1%")
    ax.axvline(CHANGE_AT,color="black",ls="--",label="mudança verdadeira")
    ax.legend();ax.set(title="Uma mudança pequena é difícil de ver ponto a ponto",xlabel="tempo",ylabel=r"$Z_t$");plt.show()
    """),
    md(r"""
    ## 4. CUSUM como razão de verossimilhança

    Para detectar mudança positiva de $N(0,1)$ para $N(\delta,1)$, a log-razão de
    verossimilhança de uma observação é

    $$\log\frac{f_{\delta}(Z_t)}{f_0(Z_t)}
    =\delta Z_t-\frac{\delta^2}{2}
    =\delta\left(Z_t-\frac{\delta}{2}\right).$$

    Removendo o fator positivo $\delta$, o CUSUM unilateral de Page é

    $$C_t^+=\max\{0,C_{t-1}^++Z_t-k\},\qquad k=\delta/2.$$

    O máximo com zero esquece evidência acumulada na direção contrária. Para mudança negativa,

    $$C_t^-=\min\{0,C_{t-1}^-+Z_t+k\}.$$

    Alertamos quando $C_t^+\ge h$ ou $C_t^-\le-h$. O parâmetro $k$ define a mudança-alvo e $h$
    controla o compromisso entre falso alarme e atraso.
    """),
    code("""
    @dataclass(frozen=True)
    class CusumResult:
        positive:np.ndarray
        negative:np.ndarray
        alarm:np.ndarray
        first_alarm:int|None

    def two_sided_cusum(z,*,reference,threshold):
        z=np.asarray(z);positive=np.empty(len(z));negative=np.empty(len(z));cp=cn=0.0
        for t,value in enumerate(z):
            cp=max(0.,cp+value-reference);cn=min(0.,cn+value+reference)
            positive[t]=cp;negative[t]=cn
        alarm=(positive>=threshold)|(negative<=-threshold)
        indices=np.flatnonzero(alarm)
        return CusumResult(positive,negative,alarm,int(indices[0]) if len(indices) else None)

    cusum=two_sided_cusum(surprises,reference=.4,threshold=5)
    pd.Series({"primeiro alerta":cusum.first_alarm,
        "falso antes da mudança":cusum.first_alarm is not None and cusum.first_alarm<CHANGE_AT,
        "atraso":None if cusum.first_alarm is None else cusum.first_alarm-CHANGE_AT})
    """),
    code("""
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    axes[0].plot(surprises,color=COLORS["gray"],lw=.8);axes[0].axvline(CHANGE_AT,color="black",ls="--")
    axes[1].plot(cusum.positive,color=COLORS["blue"],label=r"$C_t^+$")
    axes[1].plot(cusum.negative,color=COLORS["purple"],label=r"$C_t^-$")
    axes[1].axhline(5,color=COLORS["red"],ls="--");axes[1].axhline(-5,color=COLORS["red"],ls="--")
    axes[1].axvline(CHANGE_AT,color="black",ls="--")
    if cusum.first_alarm is not None:axes[1].axvline(cusum.first_alarm,color=COLORS["orange"],label="primeiro alerta")
    axes[1].legend();axes[1].set(xlabel="tempo",ylabel="CUSUM");plt.tight_layout();plt.show()
    """),
    md(r"""
    ## 5. EWMA: memória exponencial

    O EWMA atualiza

    $$E_t=\lambda Z_t+(1-\lambda)E_{t-1},\qquad E_0=0.$$

    Expandindo a recorrência,

    $$E_t=\lambda\sum_{j=0}^{t-1}(1-\lambda)^jZ_{t-j}.$$

    Sob $Z_t\overset{iid}{\sim}N(0,1)$,

    $$\operatorname{Var}(E_t)=
    \frac{\lambda}{2-\lambda}\left[1-(1-\lambda)^{2t}\right].$$

    Portanto, limites transitórios são

    $$\pm L\sqrt{\frac{\lambda}{2-\lambda}
    [1-(1-\lambda)^{2t}]}.$$

    Limites constantes de regime permanente são mais largos no início. A escolha entre limites
    transitórios e steady-state faz parte da política e altera o falso alarme inicial.
    """),
    code("""
    @dataclass(frozen=True)
    class EwmaResult:
        statistic:np.ndarray
        lower:np.ndarray
        upper:np.ndarray
        alarm:np.ndarray
        first_alarm:int|None

    def ewma_chart(z,*,smoothing,limit_multiplier):
        z=np.asarray(z);ewma=np.empty(len(z));previous=0.0
        for t,value in enumerate(z):
            previous=smoothing*value+(1-smoothing)*previous;ewma[t]=previous
        steps=np.arange(1,len(z)+1)
        variance=smoothing/(2-smoothing)*(1-(1-smoothing)**(2*steps))
        upper=limit_multiplier*np.sqrt(variance);lower=-upper
        alarm=(ewma>=upper)|(ewma<=lower);indices=np.flatnonzero(alarm)
        return EwmaResult(ewma,lower,upper,alarm,int(indices[0]) if len(indices) else None)

    ewma=ewma_chart(surprises,smoothing=.15,limit_multiplier=3.2)
    pd.Series({"primeiro alerta":ewma.first_alarm,
        "falso antes da mudança":ewma.first_alarm is not None and ewma.first_alarm<CHANGE_AT,
        "atraso":None if ewma.first_alarm is None else ewma.first_alarm-CHANGE_AT})
    """),
    code("""
    fig,ax=plt.subplots(figsize=(12,4.5));ax.plot(ewma.statistic,color=COLORS["blue"],label="EWMA")
    ax.plot(ewma.upper,color=COLORS["red"],ls="--");ax.plot(ewma.lower,color=COLORS["red"],ls="--")
    ax.axvline(CHANGE_AT,color="black",ls="--",label="mudança")
    if ewma.first_alarm is not None:ax.axvline(ewma.first_alarm,color=COLORS["orange"],label="primeiro alerta")
    ax.legend();ax.set(title="EWMA e limites transitórios",xlabel="tempo");plt.show()
    """),
    md(r"""
    ## 6. CUSUM e EWMA não são equivalentes

    CUSUM acumula evidência orientada por uma mudança-alvo $delta=2k$ e reinicia evidência
    contrária. EWMA é um estimador local suavizado cuja memória efetiva depende de $\lambda$.

    - $lambda$ pequeno enfatiza mudanças pequenas e persistentes, mas responde lentamente;
    - $lambda$ grande acompanha mudanças rápidas e também mais ruído;
    - $k$ pequeno torna o CUSUM sensível a mudanças menores;
    - $h$ ou $L$ maiores reduzem falso alarme e aumentam atraso.

    O detector e seus parâmetros precisam corresponder ao tipo e ao custo da mudança relevante.
    """),
    md(r"""
    ## 7. Run length e métricas sequenciais

    Defina o tempo de alarme $T_A=\inf\{t:\text{estatística cruza o limite}\}$.

    - **ARL$_0=E_0[T_A]$:** tempo médio até falso alarme sob estabilidade;
    - **ARL$_1$:** tempo médio até alarme quando a mudança existe desde o início;
    - **falso alarme antes de $\nu$:** $P(T_A\le\nu)$;
    - **atraso:** $T_A-\nu$, condicionado a não alertar antes e detectar após a mudança;
    - **censura:** nenhum alerta até o fim do horizonte observado.

    Reportar apenas o atraso das execuções bem-sucedidas pode esconder detectores que raramente
    detectam. Por isso atraso, probabilidade de detecção, falso alarme e censura devem aparecer
    juntos.
    """),
    code("""
    def first_run_length(first_alarm,max_length):
        return max_length+1 if first_alarm is None else first_alarm+1

    def cusum_run_lengths(repetitions,max_length,*,reference,threshold,shift=0,change_at=0,seed=0):
        rng=np.random.default_rng(seed);runs=np.empty(repetitions,dtype=int)
        for b in range(repetitions):
            z=np.r_[rng.normal(size=change_at),rng.normal(loc=shift,size=max_length-change_at)]
            result=two_sided_cusum(z,reference=reference,threshold=threshold)
            runs[b]=first_run_length(result.first_alarm,max_length)
        return runs

    def ewma_run_lengths(repetitions,max_length,*,smoothing,L,shift=0,change_at=0,seed=0):
        rng=np.random.default_rng(seed);runs=np.empty(repetitions,dtype=int)
        for b in range(repetitions):
            z=np.r_[rng.normal(size=change_at),rng.normal(loc=shift,size=max_length-change_at)]
            result=ewma_chart(z,smoothing=smoothing,limit_multiplier=L)
            runs[b]=first_run_length(result.first_alarm,max_length)
        return runs

    def summarize_runs(runs,change_at,max_length):
        runs=np.asarray(runs);false=runs<=change_at
        detected=(runs>change_at)&(runs<=max_length);delays=runs[detected]-change_at
        return {"falso alarme":false.mean(),"detecção após mudança":detected.mean(),
            "atraso médio | detectou":delays.mean() if len(delays) else np.nan,
            "censura":(runs>max_length).mean()}
    """),
    md(r"""
    ## 8. Calibração por Monte Carlo

    Não escolheremos $h$ e $L$ olhando a trajetória que queremos detectar. Primeiro simulamos o
    regime estável e medimos falso alarme dentro de uma janela operacional. Depois congelamos os
    parâmetros e avaliamos mudanças alternativas.

    A tabela usa o mesmo horizonte e as mesmas sementes para tornar comparações menos ruidosas. Os
    números estimam desempenho sob $N(0,1)$ independente; não são garantias para dados reais.
    """),
    code("""
    calibration=[]
    for threshold in [4.,5.,6.]:
        runs=cusum_run_lengths(600,400,reference=.4,threshold=threshold,seed=1202)
        calibration.append({"detector":f"CUSUM h={threshold}","P(falso em 400)":np.mean(runs<=400),
            "run length médio censurado":runs.mean()})
    for L in [3.,3.3,3.6]:
        runs=ewma_run_lengths(600,400,smoothing=.15,L=L,seed=1202)
        calibration.append({"detector":f"EWMA L={L}","P(falso em 400)":np.mean(runs<=400),
            "run length médio censurado":runs.mean()})
    pd.DataFrame(calibration).round(3)
    """),
    md(r"""
    ## 9. Avaliação depois de congelar a política

    Fixaremos CUSUM $(k=0{,}4,h=6)$ e EWMA $(\lambda=0{,}15,L=3{,}6)$. A mudança ocorre em 150 e o
    horizonte termina em 350. Avaliamos $delta\in\{0{,}3,0{,}6,1{,}0\}$ com novas sementes.
    """),
    code("""
    rows=[]
    for shift in [.3,.6,1.0]:
        c_runs=cusum_run_lengths(800,350,reference=.4,threshold=6,shift=shift,change_at=150,seed=1210+int(10*shift))
        e_runs=ewma_run_lengths(800,350,smoothing=.15,L=3.6,shift=shift,change_at=150,seed=1220+int(10*shift))
        for name,runs in [("CUSUM",c_runs),("EWMA",e_runs)]:
            row={"detector":name,"mudança":shift};row.update(summarize_runs(runs,150,350));rows.append(row)
    performance=pd.DataFrame(rows)
    performance.round(3)
    """),
    md(r"""
    **Leitura:** mudanças maiores tendem a ser detectadas mais cedo e com maior probabilidade. A
    taxa de falso alarme não deve variar muito com $delta$ porque é determinada pelo trecho
    anterior à mudança. Se escolhermos o detector depois de ver esta tabela, ela deixa de ser uma
    avaliação final e vira parte da calibração.
    """),
    md(r"""
    ## 10. Experimento interativo em uma trajetória fixa

    O gráfico ajuda a compreender parâmetros, mas não substitui Monte Carlo. A mesma trajetória pode
    favorecer acidentalmente uma configuração.
    """),
    code("""
    def explore_detectors(k=.4,h=6.,smoothing=.15,L=3.6):
        c=two_sided_cusum(surprises,reference=k,threshold=h)
        e=ewma_chart(surprises,smoothing=smoothing,limit_multiplier=L)
        fig,axes=plt.subplots(1,2,figsize=(14,4))
        axes[0].plot(c.positive,color=COLORS["blue"]);axes[0].plot(c.negative,color=COLORS["purple"])
        axes[0].axhline(h,color=COLORS["red"],ls="--");axes[0].axhline(-h,color=COLORS["red"],ls="--")
        axes[0].axvline(CHANGE_AT,color="black",ls="--");axes[0].set_title(f"CUSUM; alerta={c.first_alarm}")
        axes[1].plot(e.statistic,color=COLORS["blue"]);axes[1].plot(e.upper,color=COLORS["red"],ls="--")
        axes[1].plot(e.lower,color=COLORS["red"],ls="--");axes[1].axvline(CHANGE_AT,color="black",ls="--")
        axes[1].set_title(f"EWMA; alerta={e.first_alarm}");plt.tight_layout();plt.show()
    widgets.interact(explore_detectors,
        k=widgets.FloatSlider(min=.05,max=1,step=.05,value=.4),
        h=widgets.FloatSlider(min=2,max=10,step=.5,value=6),
        smoothing=widgets.FloatSlider(min=.03,max=.8,step=.03,value=.15),
        L=widgets.FloatSlider(min=2,max=5,step=.1,value=3.6));
    """),
    md(r"""
    ## 11. Quando a entrada não é $N(0,1)$ independente

    Se $widehat\sigma_t$ estiver subestimado, $Z_t$ fica excessivamente disperso e alarmes se
    multiplicam. Se houver autocorrelação positiva, evidências consecutivas não são independentes e
    CUSUM/EWMA acumulam padrões que o modelo deveria ter previsto. Caudas pesadas também aumentam
    cruzamentos.

    Portanto, antes do detector:

    1. preserve forecasts e calcule resíduos de um passo;
    2. verifique média, escala, ACF e estabilidade dos resíduos;
    3. calibre com simulação ou bootstrap que preserve a dependência residual;
    4. monitore também a calibração do modelo, não apenas o processo observado.

    Um alarme pode significar mudança no sistema, falha de dados ou má especificação do forecast.
    """),
    code("""
    rng=np.random.default_rng(1230)
    iid=rng.normal(size=5000)
    ar=np.empty(5000);ar[0]=iid[0]
    for t in range(1,len(ar)):ar[t]=.65*ar[t-1]+np.sqrt(1-.65**2)*iid[t]
    iid_runs=[];ar_runs=[]
    for start in range(0,5000,250):
        iid_runs.append(first_run_length(two_sided_cusum(iid[start:start+250],reference=.4,threshold=6).first_alarm,250))
        ar_runs.append(first_run_length(two_sided_cusum(ar[start:start+250],reference=.4,threshold=6).first_alarm,250))
    pd.Series({"blocos iid com alarme":np.mean(np.array(iid_runs)<=250),
        "blocos AR(1) com alarme":np.mean(np.array(ar_runs)<=250)}).round(3)
    """),
    md(r"""
    ## 12. Política de alerta além do detector

    Cruzar um limiar não define sozinho uma ação. Uma política operacional precisa declarar:

    - detector, parâmetros, versão e população monitorada;
    - regra de reset ou continuidade após o alarme;
    - cooldown e deduplicação de alertas;
    - severidade, responsável e evidência anexada;
    - condições de retreino e de encerramento;
    - custo de falso positivo, falso negativo e atraso.

    O detector produz evidência estatística. A política transforma essa evidência em um evento
    auditável; ela não deve ser inventada retroativamente depois de conhecer o desfecho.
    """),
    md(r"""
    ## 13. Exercícios

    1. Derive a log-razão de verossimilhança entre $N(0,1)$ e $N(\delta,1)$.
    2. Mostre por que $k=\delta/2$ torna o incremento esperado positivo após a mudança-alvo.
    3. Derive a variância transitória do EWMA pela soma geométrica dos pesos ao quadrado.
    4. Calibre $h$ para obter menos de 5% de falsos alarmes em 500 períodos.
    5. Compare CUSUM com $k=0{,}15$ e $k=0{,}75$ para mudanças de 0,3 e 1,5.
    6. Refaça a calibração com inovações Student-t.
    7. Explique por que ARL estimado com censura é apenas um limite/resumo truncado.
    8. Proponha política de reset e cooldown para evitar uma sequência de alertas duplicados.
    """),
    md(r"""
    <details><summary><strong>Gabarito orientativo</strong></summary>

    1. Expanda os logaritmos das duas densidades; os termos $Z_t^2$ se cancelam.
    2. Sob média $delta$, $E[Z_t-k]=\delta-delta/2>0$.
    3. Some $\lambda^2(1-\lambda)^{2j}$ de $j=0$ a $t-1$.
    4. Simule sob o nulo, aumente $h$ e congele o menor valor que satisfaz o requisito.
    5. Referência pequena favorece mudança pequena; grande ignora oscilações menores.
    6. O limiar normalmente aumenta por causa das caudas.
    7. Valores sem alarme são registrados como $H+1$, não como tempos de alarme observados.
    8. Registre o primeiro cruzamento, investigue, reinicie somente por regra e aplique cooldown.

    </details>
    """),
    code("""
    np.testing.assert_allclose(ewma.upper,-ewma.lower)
    assert cusum.first_alarm is not None and cusum.first_alarm>=CHANGE_AT
    assert ewma.first_alarm is not None and ewma.first_alarm>=CHANGE_AT
    assert set(performance.detector)=={"CUSUM","EWMA"}
    assert np.all(performance["falso alarme"].between(0,1))
    assert np.all(performance["detecção após mudança"].between(0,1))
    assert performance.loc[performance.mudança==1.0,"atraso médio | detectou"].max() < 30
    print("Testes rápidos do Notebook 12 passaram.")
    """),
    md(r"""
    ## Síntese

    CUSUM acumula log-evidência direcionada a uma mudança; EWMA estima uma média local com memória
    exponencial. Ambos precisam de calibração sequencial. ARL$_0$, falso alarme em uma janela,
    probabilidade de detecção, atraso e censura respondem perguntas diferentes e devem ser
    reportados conjuntamente.

    **Armadilhas:** repetir testes pontuais; ajustar limiar na trajetória avaliada; ignorar
    autocorrelação; confundir surpresa do modelo com causa operacional; usar residual revisado com o
    futuro; e deixar reset/cooldown implícitos.

    O Notebook 13 integrará forecasting, artefatos imutáveis, avaliação prequential, detectores,
    risco futuro e política de alertas em um fluxo auditável.
    """),
]


if __name__ == "__main__":
    write_notebook(PATH, CELLS, 12)
