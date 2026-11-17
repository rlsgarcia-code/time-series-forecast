from pathlib import Path

from notebook_builder import code, md, setup_cell, write_notebook

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "08_sistema_integrado.ipynb"

CELLS = [
    md(r"""
    # Notebook 08 — Sistema integrado de forecasting

    **Do experimento ao fluxo reproduzível, auditável e monitorado**

    Este capítulo integra contratos de dados, modelos candidatos, backtesting, seleção, artefatos,
    previsão probabilística, avaliação prequential, risco e alertas. O objetivo não é criar um
    serviço de produção completo, mas uma proposta executável com fronteiras claras.
    """),
    md(r"""
    ## Objetivos e arquitetura

    Ao final, você poderá executar:

    `dados → contrato → candidatos → backtest → seleção → fit final → artefato → monitoramento`.

    O sistema preserva previsões antes da chegada do realizado, separa pesquisa de auditoria e
    distingue anomalia individual, drift persistente e risco futuro.
    """),
    md(r"""
    ## Mapa conceitual: separar responsabilidades para preservar evidência

    O sistema é uma cadeia de contratos, não uma função gigante. Cada fronteira responde uma
    pergunta e produz um artefato verificável:

    | Componente | Responsabilidade | Não deve fazer |
    |---|---|---|
    | contrato de dados | validar significado e disponibilidade | corrigir silenciosamente dados |
    | modelo | mapear histórico para distribuição futura | escolher a si próprio |
    | backtester | reproduzir informação em cada origem | enxergar realizado antes da previsão |
    | selector | aplicar regra previamente definida | ajustar regra ao teste final |
    | artefato | congelar previsão, cutoff e metadados | mudar após observar o futuro |
    | monitor | comparar previsão preservada com realizado | reescrever previsão passada |

    Essa separação permite auditar se uma melhora veio do modelo, de dados adicionais ou de uma
    mudança indevida no protocolo.
    """),
    md(r"""
    ## Espinha matemática do sistema

    Um modelo de forecasting pode ser visto como um mapa mensurável

    $$\mathcal M:\mathcal F_T\longmapsto
    \{P_{T,h}:h=1,\ldots,H\},$$

    onde $P_{T,h}$ é a distribuição de $Y_{T+h}$ condicionada à informação legalmente disponível
    no cutoff $T$. O backtest estima o risco prequential de um modelo $m$:

    $$\widehat R(m)=\frac{1}{|\mathcal I|H}
    \sum_{T\in\mathcal I}\sum_{h=1}^{H}
    L\bigl(P^{(m)}_{T,h},Y_{T+h}\bigr),$$

    e a seleção aplica uma regra previamente declarada, por exemplo
    $\widehat m=\arg\min_{m\in\mathcal C}\widehat R(m)$, sujeita a restrições de calibração.

    Depois da promoção, cada forecast preservado gera um resíduo padronizado

    $$z_{T,h}=\frac{Y_{T+h}-\mu_{T,h}}{\sigma_{T,h}}.$$

    Sob uma distribuição normal bem calibrada, esperamos aproximadamente $z\sim N(0,1)$. Uma
    anomalia corresponde a um evento individual de baixa probabilidade; drift corresponde a uma
    mudança persistente na distribuição dos $z$. Para um evento marginal $A_h=\{Y_{T+h}>L\}$,
    $P(A_h)$ vem de $P_{T,h}$; para um evento de trajetória
    $A=\{\max_{1\le h\le H}Y_{T+h}>L\}$, precisamos da distribuição conjunta. Com caminhos
    simulados $Y^{(b)}$, a aproximação de Monte Carlo é

    $$\widehat P(A)=\frac1B\sum_{b=1}^{B}
    \mathbf1\!\left\{\max_hY_{T+h}^{(b)}>L\right\}.$$

    Essa formulação mostra por que o sistema precisa preservar cutoff, horizonte, distribuição e
    dependência entre horizontes — não apenas uma tabela de médias.
    """),
    setup_cell("from dataclasses import asdict, dataclass\nimport json\nfrom pathlib import Path\nimport tempfile\nfrom scipy.stats import norm"),
    md(r"""
    ## 1. Contrato de dados e configuração

    O contrato torna explícitos frequência, timezone, histórico mínimo e horizonte. Em dados reais,
    também validaria timestamps duplicados, gaps, chaves e instante de corte. Falhar cedo é melhor
    que produzir uma previsão silenciosamente desalinhada.

    Frequência não é apenas metadado: “horizonte 7” significa sete dias em frequência diária e sete
    meses em frequência mensal. Timezone define a fronteira de cada período. O cutoff registra a
    última observação legalmente disponível e é essencial para provar ausência de leakage.
    """),
    code("""
    @dataclass(frozen=True)
    class SeriesContract:
        frequency: str
        timezone: str
        horizon: int
        minimum_history: int

        def validate(self,values):
            x=np.asarray(values,dtype=float)
            if x.ndim!=1 or len(x)<self.minimum_history:raise ValueError("histórico insuficiente")
            if not np.all(np.isfinite(x)) or self.horizon<1:raise ValueError("dados inválidos")
            return x

    @dataclass(frozen=True)
    class RunConfig:
        initial_train_size:int=180
        horizon:int=6
        step:int=3
        confidence:float=.95
        selection_metric:str="crps"

    contract=SeriesContract("D","UTC",horizon=6,minimum_history=120)
    config=RunConfig()
    """),
    md(r"""
    ## 2. Série de demonstração

    Geramos um AR(1) estável e introduzimos uma mudança de média somente perto do fim. A seleção
    ocorre antes da mudança; o monitor deve perceber que os erros posteriores deixaram de parecer
    compatíveis com a distribuição emitida.
    """),
    code("""
    def simulate_system_series(n=420,seed=801):
        rng=np.random.default_rng(seed);x=np.empty(n);x[0]=1.5
        for t in range(1,n):
            mean_shift=0 if t<360 else .9
            x[t]=.25+mean_shift+.78*x[t-1]+rng.normal(0,.8)
        return x

    values=contract.validate(simulate_system_series())
    fig,ax=plt.subplots(figsize=(11,4));ax.plot(values,color=COLORS["blue"]);ax.axvline(360,color=COLORS["red"],ls="--",label="mudança simulada")
    ax.set(title="Série usada no fluxo integrado",xlabel="tempo");ax.legend();plt.show()
    """),
    md(r"""
    ## 3. Interface mínima de modelos

    Todos os candidatos precisam produzir média e desvio-padrão no mesmo formato. Aqui usamos
    funções simples; em produção, uma interface/classe adicionaria versionamento, serialização e
    declaração de dependências.

    Uma interface de distribuição permite trocar a família estatística sem reescrever avaliação e
    monitoramento. O contrato precisa declarar ao menos horizonte, média/quantis ou amostras,
    cutoff e convenção de índice. Retornar apenas um ponto impediria avaliar calibração e risco.
    """),
    code("""
    def fit_ar1(x):
        y=x[1:];X=np.column_stack([np.ones(len(x)-1),x[:-1]])
        c,phi=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@np.array([c,phi])
        return {"c":c,"phi":phi,"sigma":np.sqrt(res@res/(len(y)-2))}

    def predict_ar1(history,h):
        p=fit_ar1(history);steps=np.arange(1,h+1)
        mean=p["phi"]**steps*history[-1]+p["c"]*(1-p["phi"]**steps)/(1-p["phi"])
        std=p["sigma"]*np.sqrt(np.cumsum(p["phi"]**(2*np.arange(h))))
        return mean,std,p

    def predict_naive(history,h):
        sigma=np.std(np.diff(history),ddof=1);steps=np.arange(1,h+1)
        return np.repeat(history[-1],h),sigma*np.sqrt(steps),{"sigma_diff":sigma}

    MODEL_REGISTRY={"ar1":predict_ar1,"naive":predict_naive}
    """),
    md(r"""
    ### 3.1 Por que `mean` e `std` são a interface do sistema?

    No AR(1), a recorrência estimada é generalizada para inferência multi-step por
    $\widehat m_h=\widehat\phi^hX_T+\widehat c(1-\widehat\phi^h)/(1-\widehat\phi)$ e
    $\widehat s_h^2=\widehat\sigma^2\sum_{j=0}^{h-1}\widehat\phi^{2j}$.
    O registry não precisa conhecer essa álgebra: recebe a distribuição resultante em um contrato
    comum. Isso separa três responsabilidades:

    1. o modelo transforma histórico e parâmetros em distribuição futura;
    2. o backtester verifica a generalização em dados não vistos;
    3. o monitor compara cada realizado com a distribuição preservada antes de sua chegada.

    Os intervalos continuam sendo plug-in. Um sistema de maior criticidade deve registrar também a
    incerteza de estimação, por bootstrap, ensemble ou posterior Bayesiana.
    """),
    md(r"""
    ## 4. Backtester central

    O backtester não conhece detalhes internos do modelo. Ele controla cortes temporais e registra
    uma tabela prequential imutável. Reservamos as últimas 60 observações para monitoramento;
    seleção usa somente o passado anterior.

    “Prequential” combina *predictive* e *sequential*: primeiro emitimos e armazenamos a previsão;
    somente depois chega o realizado e calculamos o score. Essa ordem operacional é a garantia
    concreta contra avaliações retrospectivas que usam informação futura.
    """),
    code("""
    def rolling_origins(n,initial,horizon,step):return list(range(initial,n-horizon+1,step))

    def backtest(values,models,config,end=None):
        end=len(values) if end is None else end;rows=[]
        for origin in rolling_origins(end,config.initial_train_size,config.horizon,config.step):
            history=values[:origin];actual=values[origin:origin+config.horizon]
            for name,predict in models.items():
                mean,std,_=predict(history,config.horizon)
                for h in range(config.horizon):rows.append({"model":name,"origin":origin,"h":h+1,
                    "actual":actual[h],"mean":mean[h],"std":std[h]})
        return pd.DataFrame(rows)

    selection_end=360
    bt=backtest(values,MODEL_REGISTRY,config,end=selection_end)
    bt.head()
    """),
    md(r"""
    ## 5. Avaliador probabilístico e seleção

    CRPS é a regra principal; cobertura e largura são guardrails. Uma seleção responsável pode
    rejeitar o menor CRPS se cobertura estiver muito distante do alvo ou se o modelo falhar em
    horizontes críticos.
    """),
    code("""
    def gaussian_crps(y,mean,std):
        z=(y-mean)/std
        return std*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/np.sqrt(np.pi))

    def score_predictions(frame,confidence=.95):
        out=frame.copy();z=norm.ppf((1+confidence)/2);error=out.actual-out["mean"]
        out["abs_error"]=np.abs(error);out["sq_error"]=error**2
        out["lower"]=out["mean"]-z*out["std"];out["upper"]=out["mean"]+z*out["std"]
        out["covered"]=(out.actual>=out.lower)&(out.actual<=out.upper)
        out["width"]=out.upper-out.lower;out["crps"]=gaussian_crps(out.actual,out["mean"],out["std"])
        return out

    scored=score_predictions(bt)
    scoreboard=scored.groupby("model").agg(MAE=("abs_error","mean"),RMSE=("sq_error",lambda x:np.sqrt(x.mean())),
        coverage=("covered","mean"),width=("width","mean"),crps=("crps","mean")).sort_values("crps")
    selected_model=scoreboard.index[0]
    scoreboard.round(4)
    """),
    md(r"""
    ## 6. Fit final e artefato versionado

    O artefato preserva cutoff, modelo, parâmetros, distribuição prevista, configuração e métricas
    de seleção. Sem cutoff e metadados, não é possível reproduzir o que o sistema sabia.

    Diferencie três objetos: **modelo** é a regra estimada; **run** é uma execução com dados e
    configuração específicos; **forecast artifact** é a previsão imutável produzida por uma run.
    Duas runs do mesmo modelo podem gerar artefatos diferentes por cutoff, versão do código ou
    histórico. A promoção deve apontar para identificadores de todos eles.
    """),
    code("""
    @dataclass(frozen=True)
    class ForecastArtifact:
        model_name:str;cutoff:int;horizon:list;mean:list;lower:list;upper:list;metadata:dict
        def save(self,path):
            Path(path).write_text(json.dumps(asdict(self),indent=2,sort_keys=True),encoding="utf-8")
        @classmethod
        def load(cls,path):return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    history=values[:selection_end]
    mean,std,params=MODEL_REGISTRY[selected_model](history,config.horizon)
    z=norm.ppf((1+config.confidence)/2)
    artifact=ForecastArtifact(selected_model,selection_end,list(range(1,config.horizon+1)),mean.tolist(),
        (mean-z*std).tolist(),(mean+z*std).tolist(),{"config":asdict(config),"parameters":params,
        "selection_scores":scoreboard.loc[selected_model].to_dict()})

    with tempfile.TemporaryDirectory() as tmp:
        artifact_path=Path(tmp)/"forecast.json";artifact.save(artifact_path);restored=ForecastArtifact.load(artifact_path)
        print(artifact_path.read_text()[:800]+"...")
    assert restored==artifact
    """),
    md(r"""
    ## 7. Monitoramento prequential

    Para cada nova observação, preservamos a previsão emitida antes dela e calculamos surpresa
    padronizada $z_t=(y_t-\mu_t)/\sigma_t$. Um ponto extremo é anomalia; vários erros com mesmo
    sinal sugerem drift. O monitor não deve reescrever previsões antigas após observar o futuro.

    Há pelo menos três falhas distintas: **qualidade de dados** (gap, atraso, unidade errada),
    **anomalia** (evento raro sob o modelo) e **drift** (mudança persistente do mecanismo). Retreinar
    imediatamente após qualquer surpresa pode fazer o modelo perseguir ruído; primeiro classificamos
    a evidência e aplicamos uma política com persistência mínima.
    """),
    code("""
    def monitor_stream(values,start,model_name):
        rows=[]
        for t in range(start,len(values)):
            mean,std,_=MODEL_REGISTRY[model_name](values[:t],1)
            surprise=(values[t]-mean[0])/std[0]
            rows.append({"t":t,"actual":values[t],"forecast":mean[0],"std":std[0],"surprise":surprise})
        return pd.DataFrame(rows)

    monitoring=monitor_stream(values,selection_end,selected_model)
    monitoring["point_alert"]=np.abs(monitoring.surprise)>2.58
    monitoring["ewma_surprise"]=monitoring.surprise.ewm(alpha=.15,adjust=False).mean()
    monitoring["drift_alert"]=np.abs(monitoring.ewma_surprise)>1.0

    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    axes[0].plot(monitoring.t,monitoring.actual,color="black",label="realizado")
    axes[0].plot(monitoring.t,monitoring.forecast,color=COLORS["orange"],label="forecast 1 passo");axes[0].legend()
    axes[1].plot(monitoring.t,monitoring.surprise,color=COLORS["blue"],alpha=.5,label="surpresa")
    axes[1].plot(monitoring.t,monitoring.ewma_surprise,color=COLORS["red"],lw=2,label="EWMA")
    axes[1].axhline(1,color="black",ls="--");axes[1].axhline(-1,color="black",ls="--");axes[1].legend()
    axes[1].set(xlabel="tempo",ylabel="erro padronizado");plt.tight_layout();plt.show()
    """),
    md(r"""
    ## 8. Risco futuro é diferente de anomalia

    Uma anomalia compara uma observação recebida com sua previsão anterior. Risco futuro calcula,
    antes do evento, $P(X_{T+h}>L)$. Para distribuição normal: $1-\Phi((L-\mu_h)/\sigma_h)$.

    Essa probabilidade é marginal em um horizonte. A chance de cruzar o limite **em qualquer um**
    dos próximos dez períodos depende da correlação entre horizontes e não é a soma das marginais.
    Para esse evento de trajetória, simulamos caminhos conjuntos e contamos a fração cujo máximo
    ultrapassa o limite.
    """),
    code("""
    risk_history=values[:-10]
    risk_mean,risk_std,_=MODEL_REGISTRY[selected_model](risk_history,10)
    threshold=6.0
    exceedance_probability=1-norm.cdf((threshold-risk_mean)/risk_std)
    risk_table=pd.DataFrame({"h":range(1,11),"média":risk_mean,"desvio":risk_std,
        f"P(X>{threshold})":exceedance_probability})
    risk_table.round(4)
    """),
    md(r"""
    ## 9. Política de atualização e auditoria

    Alertas não devem disparar retreino automático sem regra. Um fluxo seguro registra evidência,
    verifica qualidade de dados, compara janela recente com baseline e exige critério explícito
    para promover novo modelo. Toda promoção preserva versão anterior e backtest usado na decisão.
    """),
    code("""
    audit_summary=pd.Series({"modelo selecionado":selected_model,"cutoff do artefato":artifact.cutoff,
        "alertas pontuais":int(monitoring.point_alert.sum()),"alertas de drift":int(monitoring.drift_alert.sum()),
        "CRPS de seleção":scoreboard.loc[selected_model,"crps"],"cobertura de seleção":scoreboard.loc[selected_model,"coverage"]})
    audit_summary
    """),
    md(r"""
    ## 10. Exercícios

    1. Adicione validação de timestamps ao contrato. 2. Inclua AR(2) no registry sem alterar o
    backtester. 3. Defina guardrail de cobertura para seleção. 4. Diferencie anomalia e drift com
    exemplos. 5. Versione dois artefatos e compare metadados. 6. Proponha regra de retreino que não
    reaja a um único ponto. 7. Calcule probabilidade de cruzar um limite em qualquer horizonte.
    """),
    md(r"""
    <details><summary><strong>Gabarito orientativo</strong></summary>

    1. Verifique ordem, unicidade, frequência e timezone antes de extrair valores. 2. Implemente a
    mesma assinatura `(history,horizon) -> mean,std,params`. 3. Rejeite candidatos cuja cobertura
    fique fora de uma faixa pré-definida. 4. Anomalia é pontual; drift é evidência acumulada.
    5. Compare cutoff, código, dados, parâmetros e configuração. 6. Exija persistência e auditoria
    antes de promover. 7. A probabilidade conjunta exige simular caminhos, não somar marginais.
    </details>
    """),
    code("""
    assert contract.validate(values).shape==(420,)
    assert set(scoreboard.index)==set(MODEL_REGISTRY)
    assert len(artifact.mean)==config.horizon
    assert monitoring.t.iloc[0]==selection_end
    assert np.all((exceedance_probability>=0)&(exceedance_probability<=1))
    assert restored==artifact
    print("✓ Contratos, artefato, monitor e risco passaram nos testes rápidos.")
    """),
    md(r"""
    ## 11. Entrega arquitetural

    O protótipo prova os contratos essenciais:

    - dados validados antes da modelagem;
    - modelos intercambiáveis por uma interface de distribuição;
    - seleção baseada em backtest temporal preservado;
    - artefato reproduzível com cutoff e metadados;
    - avaliação prequential sem reescrever o passado;
    - separação entre anomalia observada, drift acumulado e risco futuro.

    Extensões previstas na visão do sistema: estado-espaço/Kalman, mudança de regime, change point,
    detectores CUSUM/EWMA calibrados, armazenamento persistente e execução agendada. O repositório
    agora contém a base executável para evoluir essas capacidades sem misturar pesquisa e operação.
    """),
]

if __name__ == "__main__":
    write_notebook(PATH,CELLS,8)

