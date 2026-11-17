from pathlib import Path

from notebook_builder import code, md, setup_cell, write_notebook


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "11_change_point.ipynb"


CELLS = [
    md(r"""
    # Notebook 11 — Quebras estruturais e change-point detection

    **Segmentar o passado sem transformar cada ruído em uma mudança**

    Uma quebra estrutural altera parâmetros do processo a partir de uma posição desconhecida. Neste
    capítulo construiremos detectores offline para mudanças de média, começando por uma única
    quebra e chegando à segmentação penalizada de múltiplas quebras.
    """),
    md(r"""
    ## Pré-requisitos, objetivos e roteiro

    **Pré-requisitos:** média, variância, máxima verossimilhança, programação dinâmica básica e
    distinção entre filtragem online e suavização histórica.

    Ao final, você saberá:

    1. distinguir outlier, regime recorrente, drift gradual e change point;
    2. formular o custo de um segmento gaussiano;
    3. detectar uma quebra por busca exata e por CUSUM offline;
    4. encontrar múltiplas quebras por programação dinâmica;
    5. controlar sobreajuste com penalidade e tamanho mínimo;
    6. calibrar evidência sob uma hipótese sem mudança;
    7. avaliar localizações com tolerância e matching um-para-um.
    """),
    md(r"""
    ## 1. Quatro fenômenos diferentes

    | Fenômeno | Estrutura | Pergunta adequada |
    |---|---|---|
    | outlier | uma observação rara | o ponto é compatível com o modelo atual? |
    | drift gradual | parâmetro se move continuamente | há evidência sequencial acumulada? |
    | Markov switching | regimes podem reaparecer | qual a probabilidade de cada regime agora? |
    | change point | parâmetro muda em posição desconhecida | quantos segmentos existem e onde começam? |

    Uma segmentação offline usa toda a amostra $y_{1:n}$. Ela é útil para diagnóstico histórico,
    mas a data estimada retrospectivamente não é a data em que um alerta teria sido emitido. O
    Notebook 12 tratará explicitamente detecção sequencial e atraso de alarme.
    """),
    md(r"""
    ## 2. Modelo matemático por segmentos

    Se existem quebras $0=\tau_0<\tau_1<\cdots<\tau_m<\tau_{m+1}=n$, assumimos

    $$Y_t=\mu_j+\varepsilon_t,
    \qquad \tau_j<t\le\tau_{j+1},
    \qquad \varepsilon_t\overset{iid}{\sim}N(0,\sigma^2).$$

    Para um segmento semiaberto $[a,b)$, o estimador de máxima verossimilhança da média é

    $$\widehat\mu_{a:b}=\frac{1}{b-a}\sum_{t=a}^{b-1}y_t.$$

    Com variância comum conhecida, minimizar a log-verossimilhança negativa equivale a minimizar

    $$C(a,b)=\sum_{t=a}^{b-1}(y_t-\widehat\mu_{a:b})^2.$$

    Se adicionarmos segmentos sem custo, cada observação poderá virar seu próprio segmento. A
    inferência exige uma penalidade ou um número de mudanças previamente fixado.
    """),
    setup_cell("from dataclasses import dataclass\nimport ipywidgets as widgets"),
    md(r"""
    ## 3. Série controlada com três quebras

    A simulação mantém variância constante para isolar mudanças de média. Como conhecemos a verdade,
    podemos separar erro de localização de mera aparência visual.
    """),
    code("""
    def simulate_piecewise_constant(lengths,means,*,sigma=1.0,seed=None):
        lengths=np.asarray(lengths,dtype=int);means=np.asarray(means,dtype=float)
        latent=np.repeat(means,lengths);rng=np.random.default_rng(seed)
        values=latent+rng.normal(0,sigma,len(latent))
        breakpoints=tuple(np.cumsum(lengths)[:-1])
        return values,latent,breakpoints

    values,latent_mean,TRUE_BREAKS=simulate_piecewise_constant(
        [120,140,130,110],[0.0,2.2,-1.3,1.4],sigma=.7,seed=1101)
    fig,ax=plt.subplots(figsize=(13,5));ax.plot(values,color=COLORS["gray"],alpha=.65,label="observado")
    ax.plot(latent_mean,color="black",lw=2,label="média verdadeira")
    for point in TRUE_BREAKS:ax.axvline(point,color=COLORS["red"],ls="--")
    ax.legend();ax.set(title="Processo com médias constantes por segmento",xlabel="tempo");plt.show()
    """),
    md(r"""
    ## 4. Custo em tempo constante com somas prefixadas

    Recalcular cada média para cada candidato seria caro. Defina

    $$S_t=\sum_{i=0}^{t-1}y_i,\qquad Q_t=\sum_{i=0}^{t-1}y_i^2.$$

    Então, para $n_{ab}=b-a$,

    $$C(a,b)=(Q_b-Q_a)-\frac{(S_b-S_a)^2}{n_{ab}}.$$

    Após construir $S$ e $Q$ em $O(n)$, qualquer custo de segmento é calculado em $O(1)$. Essa
    identidade tornará viável a programação dinâmica posterior.
    """),
    code("""
    def prefix_sums(y):
        y=np.asarray(y,dtype=float)
        return np.r_[0.,np.cumsum(y)],np.r_[0.,np.cumsum(y**2)]

    def segment_sse(total,squares,start,end):
        n=end-start;segment_sum=total[end]-total[start]
        return max((squares[end]-squares[start])-segment_sum**2/n,0.0)

    S,Q=prefix_sums(values)
    direct=np.sum((values[120:260]-values[120:260].mean())**2)
    pd.Series({"SSE direta":direct,"SSE por prefixos":segment_sse(S,Q,120,260)}).round(8)
    """),
    md(r"""
    ## 5. Uma única mudança: busca exata

    Para cada candidato $\tau\in\{h,\ldots,n-h\}$, com tamanho mínimo $h$, calculamos

    $$J(\tau)=C(0,\tau)+C(\tau,n).$$

    A estimativa é $\widehat\tau=\arg\min_\tau J(\tau)$. A redução

    $$\Delta C=C(0,n)-J(\widehat\tau)$$

    mede quanto a melhor divisão melhora o ajuste, mas não é sozinha um p-valor: escolhemos o maior
    ganho entre muitas posições e precisamos considerar essa busca na calibração.
    """),
    code("""
    @dataclass(frozen=True)
    class SingleChangeResult:
        breakpoint:int
        no_change_cost:float
        split_cost:float
        improvement:float
        profile:np.ndarray

    def best_single_change(y,min_size=20):
        y=np.asarray(y);S,Q=prefix_sums(y);profile=np.full(len(y)+1,np.nan)
        for split in range(min_size,len(y)-min_size+1):
            profile[split]=segment_sse(S,Q,0,split)+segment_sse(S,Q,split,len(y))
        breakpoint=int(np.nanargmin(profile));base=segment_sse(S,Q,0,len(y))
        return SingleChangeResult(breakpoint,base,profile[breakpoint],
            base-profile[breakpoint],profile)

    single=best_single_change(values,min_size=30)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    axes[0].plot(values,color=COLORS["gray"]);axes[0].axvline(single.breakpoint,color=COLORS["red"])
    axes[0].set_title(f"Melhor divisão única: tau={single.breakpoint}")
    axes[1].plot(single.profile,color=COLORS["blue"]);axes[1].axvline(single.breakpoint,color=COLORS["red"])
    axes[1].set(title="Custo para cada posição candidata",xlabel="posição da quebra")
    plt.tight_layout();plt.show()
    """),
    md(r"""
    **Armadilha:** a série possui três quebras, mas forçamos apenas uma. O algoritmo retorna a
    divisão que mais reduz o custo global — não necessariamente a primeira quebra verdadeira. Um
    modelo incorreto ainda produz uma resposta numérica precisa.
    """),
    md(r"""
    ## 6. CUSUM offline para uma mudança de média

    Sob estabilidade, as somas dos desvios em relação à média global oscilam em torno de zero:

    $$G_t=\frac{1}{s\sqrt n}\sum_{i=1}^{t}(Y_i-\bar Y),
    \qquad G_n=0.$$

    Uma mudança de média cria uma trajetória sistemática e a posição de
    $\max_t|G_t|$ sugere a quebra. A escala $s\sqrt n$ permite comparar amostras, mas o limiar
    depende das hipóteses de independência e estabilidade da variância.
    """),
    code("""
    def cusum_mean(y):
        y=np.asarray(y);scale=np.std(y,ddof=1)
        return np.cumsum(y-y.mean())/(scale*np.sqrt(len(y)))

    one_change,_,one_true=simulate_piecewise_constant([220,280],[0,1.4],sigma=1,seed=1102)
    cusum=cusum_mean(one_change);cusum_location=int(np.argmax(np.abs(cusum)))+1
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
    axes[0].plot(one_change,color=COLORS["gray"]);axes[0].axvline(one_true[0],color="black",ls="--")
    axes[1].plot(np.arange(1,len(cusum)+1),cusum,color=COLORS["purple"])
    axes[1].axvline(cusum_location,color=COLORS["red"],label=f"máximo em {cusum_location}")
    axes[1].set(xlabel="tempo",ylabel=r"$G_t$");axes[1].legend();plt.tight_layout();plt.show()
    """),
    md(r"""
    ## 7. Calibração sob a hipótese nula

    Um valor máximo grande pode ocorrer por acaso porque examinamos muitos tempos. Simularemos
    séries sem mudança com o mesmo tamanho e calcularemos
    $M=\max_t|G_t|$. O quantil 95% aproxima um limiar de falso positivo de 5% **sob o modelo nulo
    simulado**.

    Essa calibração não é universal: autocorrelação, caudas pesadas ou heterocedasticidade mudam a
    distribuição nula. Em dados dependentes, seria necessário bootstrap em blocos ou um modelo para
    os resíduos.
    """),
    code("""
    def calibrate_cusum_threshold(n,repetitions=1500,alpha=.05,seed=0):
        rng=np.random.default_rng(seed);maxima=np.empty(repetitions)
        for b in range(repetitions):
            sample=rng.normal(size=n);maxima[b]=np.max(np.abs(cusum_mean(sample)))
        return np.quantile(maxima,1-alpha),maxima

    threshold,null_maxima=calibrate_cusum_threshold(len(one_change),seed=1103)
    observed_max=np.max(np.abs(cusum))
    pd.Series({"máximo observado":observed_max,"limiar nulo 95%":threshold,
        "p-valor Monte Carlo aproximado":(1+np.sum(null_maxima>=observed_max))/(1+len(null_maxima))}).round(4)
    """),
    md(r"""
    ## 8. Múltiplas mudanças por programação dinâmica

    Procurar todas as combinações de quebras é exponencial. A aditividade do custo permite definir

    $$F(t)=\min_{s\le t-h}\{F(s)+C(s,t)+\beta\},$$

    onde $F(t)$ é o menor custo até $t$, $h$ é o tamanho mínimo e $\beta$ penaliza cada quebra.
    Usamos $F(0)=-\beta$ para que o primeiro segmento não pague penalidade. Guardando o melhor
    predecessor de cada $t$, reconstruímos as quebras por retrocesso.

    A implementação exata abaixo custa $O(n^2)$. Algoritmos como PELT usam poda para acelerar casos
    maiores, mantendo a mesma lógica de custo aditivo e penalização.
    """),
    code("""
    @dataclass(frozen=True)
    class SegmentationResult:
        breakpoints:tuple
        objective:float
        fitted:np.ndarray

    def optimal_segmentation(y,*,penalty,min_size=20):
        y=np.asarray(y);n=len(y);S,Q=prefix_sums(y)
        objective=np.full(n+1,np.inf);previous=np.full(n+1,-1,dtype=int);objective[0]=-penalty
        for end in range(min_size,n+1):
            starts=[0]+list(range(min_size,end-min_size+1))
            starts=[start for start in starts if np.isfinite(objective[start])]
            costs=np.array([objective[start]+segment_sse(S,Q,start,end)+penalty for start in starts])
            winner=int(np.argmin(costs));objective[end]=costs[winner];previous[end]=starts[winner]
        points=[];end=n
        while previous[end]>0:points.append(previous[end]);end=previous[end]
        points=tuple(sorted(points));fitted=np.empty(n);bounds=(0,*points,n)
        for start,end in zip(bounds[:-1],bounds[1:]):fitted[start:end]=y[start:end].mean()
        return SegmentationResult(points,float(objective[n]),fitted)

    segmentation=optimal_segmentation(values,penalty=12,min_size=30)
    pd.Series({"quebras verdadeiras":TRUE_BREAKS,"quebras estimadas":segmentation.breakpoints,
        "número de segmentos":len(segmentation.breakpoints)+1})
    """),
    code("""
    fig,ax=plt.subplots(figsize=(13,5));ax.plot(values,color=COLORS["gray"],alpha=.55,label="observado")
    ax.plot(latent_mean,color="black",lw=2,label="média verdadeira")
    ax.plot(segmentation.fitted,color=COLORS["orange"],lw=2,label="segmentação estimada")
    for point in segmentation.breakpoints:ax.axvline(point,color=COLORS["red"],ls="--",alpha=.8)
    ax.legend();ax.set(title="Programação dinâmica com penalidade",xlabel="tempo");plt.show()
    """),
    md(r"""
    ## 9. Penalidade e tamanho mínimo são hipóteses

    - $eta$ pequeno recompensa ajuste e cria muitas quebras.
    - $eta$ grande favorece estabilidade e pode omitir mudanças reais.
    - `min_size` define a menor duração admissível de um mecanismo.

    Nenhum desses valores deve ser escolhido olhando apenas para a segmentação mais agradável. Eles
    representam custo de complexidade, escala do ruído e resolução temporal relevante. BIC, dados
    simulados, estabilidade e desempenho downstream fornecem critérios mais defensáveis.
    """),
    code("""
    def explore_penalty(penalty=12,min_size=30):
        result=optimal_segmentation(values,penalty=penalty,min_size=min_size)
        fig,ax=plt.subplots(figsize=(11,4));ax.plot(values,color=COLORS["gray"],alpha=.45)
        ax.plot(result.fitted,color=COLORS["orange"],lw=2)
        for point in result.breakpoints:ax.axvline(point,color=COLORS["red"],ls="--")
        ax.set_title(f"penalidade={penalty:.1f}; min_size={min_size}; mudanças={len(result.breakpoints)}")
        plt.show()
    widgets.interact(explore_penalty,
        penalty=widgets.FloatSlider(min=0,max=80,step=2,value=12),
        min_size=widgets.IntSlider(min=10,max=80,step=5,value=30));
    """),
    md(r"""
    ## 10. Incerteza da localização

    O mínimo de $J(\tau)$ é uma estimativa pontual. Um perfil quase plano indica que várias posições
    explicam os dados de maneira semelhante. Um intervalo de confiança formal depende da teoria ou
    de bootstrap apropriado; escolher arbitrariamente todos os pontos “perto do mínimo” não gera
    automaticamente cobertura válida.

    Como diagnóstico visual, normalizamos a perda relativa

    $$R(\tau)=J(\tau)-J(\widehat\tau).$$

    Quanto mais rapidamente $R$ cresce ao redor do mínimo, mais localizada está a evidência.
    """),
    code("""
    single_clean=best_single_change(one_change,min_size=40)
    relative=single_clean.profile-single_clean.split_cost
    finite=np.isfinite(relative)
    near=np.flatnonzero(finite & (relative<=np.nanquantile(relative[finite],.10)))
    fig,ax=plt.subplots(figsize=(11,4));ax.plot(relative,color=COLORS["blue"])
    ax.axvline(single_clean.breakpoint,color=COLORS["red"],label="mínimo")
    ax.set(title=f"Perfil relativo; região diagnóstica {near.min()}–{near.max()}",
        xlabel="candidato",ylabel=r"$R(\tau)$");ax.legend();plt.show()
    """),
    md(r"""
    ## 11. Avaliação requer tolerância e matching

    Exigir igualdade exata entre quebra verdadeira e estimada é excessivamente rígido; considerar
    qualquer ponto próximo como acerto múltiplo é excessivamente permissivo. Usaremos matching
    um-para-um: cada estimativa e cada verdade participam de no máximo um par dentro da tolerância.

    $$\text{precision}=\frac{TP}{\#\text{estimadas}},\qquad
    \text{recall}=\frac{TP}{\#\text{verdadeiras}},\qquad
    F_1=\frac{2PR}{P+R}.$$
    """),
    code("""
    def match_breakpoints(truth,estimates,tolerance):
        available=set(range(len(truth)));matches=[]
        for estimate in estimates:
            candidates=[i for i in available if abs(truth[i]-estimate)<=tolerance]
            if candidates:
                winner=min(candidates,key=lambda i:abs(truth[i]-estimate))
                available.remove(winner);matches.append((truth[winner],estimate))
        precision=len(matches)/len(estimates) if estimates else float(not truth)
        recall=len(matches)/len(truth) if truth else float(not estimates)
        f1=2*precision*recall/(precision+recall) if precision+recall else 0
        return {"matches":matches,"precision":precision,"recall":recall,"F1":f1}

    match_breakpoints(TRUE_BREAKS,segmentation.breakpoints,tolerance=8)
    """),
    md(r"""
    ## 12. Outliers e dependência serial

    A soma de quadrados é sensível a valores extremos: um outlier pode justificar duas quebras
    próximas para isolá-lo. Custos robustos, pré-validação de dados e penalidade ajudam, mas mudam a
    função objetivo. Da mesma forma, autocorrelação reduz o número efetivo de observações e pode
    produzir segmentos espúrios se tratarmos resíduos como independentes.

    Uma análise responsável especifica o que muda — média, variância, coeficiente AR ou toda a
    distribuição — e verifica resíduos dentro de cada segmento. “Detectar qualquer mudança” não é
    um modelo estatístico bem definido.
    """),
    md(r"""
    ## 13. Exercícios

    1. Derive a expressão prefixada do SSE expandindo os quadrados.
    2. Explique por que a melhor quebra sempre reduz ou mantém o custo dentro da amostra.
    3. Simule uma série sem mudança e examine a segmentação quando $eta=0$.
    4. Adicione um outlier e compare penalidades 5, 20 e 60.
    5. Calibre o máximo CUSUM para ruído Student-t e compare com o limiar normal.
    6. Mostre por que uma quebra em variância pode não ser bem detectada pelo custo de média.
    7. Implemente custo de variância ou log-verossimilhança gaussiana por segmento.
    8. Explique por que a data offline estimada não é o atraso de um detector online.
    """),
    md(r"""
    <details><summary><strong>Gabarito orientativo</strong></summary>

    1. Use $\sum(y_i-\bar y)^2=\sum y_i^2-(\sum y_i)^2/n$.
    2. A opção de não dividir continua disponível; otimizar sobre mais candidatos não aumenta o
       menor custo.
    3. Sem penalidade, o método cria o maior número permitido de segmentos.
    4. Penalidade pequena tende a isolar o outlier com duas quebras.
    5. Caudas pesadas elevam os quantis nulos e exigem limiar maior.
    6. Médias iguais antes/depois mantêm pequeno o ganho de SSE de média.
    7. Estime média e variância no segmento e use a log-verossimilhança maximizada.
    8. A análise offline usa observações posteriores; o detector online só conhece o prefixo atual.

    </details>
    """),
    code("""
    np.testing.assert_allclose(segment_sse(S,Q,120,260),direct)
    assert abs(cusum_location-one_true[0])<=10
    assert observed_max>threshold
    assert len(segmentation.breakpoints)==len(TRUE_BREAKS)
    assert all(abs(a-b)<=5 for a,b in zip(segmentation.breakpoints,TRUE_BREAKS))
    metrics=match_breakpoints(TRUE_BREAKS,segmentation.breakpoints,tolerance=8)
    assert metrics["precision"]==metrics["recall"]==metrics["F1"]==1
    assert abs(cusum[-1])<1e-12
    print("Testes rápidos do Notebook 11 passaram.")
    """),
    md(r"""
    ## Síntese

    Change-point detection offline transforma uma série em segmentos ao equilibrar ajuste e
    complexidade. CUSUM localiza evidência de uma mudança; custos prefixados tornam buscas rápidas;
    programação dinâmica resolve a segmentação global; penalidade e tamanho mínimo expressam
    hipóteses; e avaliação exige tolerância sem contar múltiplos acertos para a mesma quebra.

    **Armadilhas:** interpretar segmentação como causalidade; calibrar sob normalidade quando os
    resíduos são dependentes; confundir outlier com mudança; usar o futuro e relatar a detecção como
    online; ou procurar mudanças sem dizer qual parâmetro pode mudar.

    O Notebook 12 adotará a perspectiva sequencial: CUSUM e EWMA processarão uma observação por vez,
    com métricas de falso alarme e atraso de detecção.
    """),
]


if __name__ == "__main__":
    write_notebook(PATH, CELLS, 11)
