from pathlib import Path

from notebook_builder import code, md, setup_cell, write_notebook

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "07_backtesting_selecao.ipynb"

CELLS = [
    md(r"""
    # Notebook 07 — Backtesting e seleção probabilística

    **Comparar modelos sob o mesmo protocolo, sem olhar o futuro**

    Um modelo não é bom em abstrato: ele é adequado para uma série, horizonte, informação
    disponível e função de perda. Construiremos rolling-origin, métricas pontuais e probabilísticas,
    calibração, sharpness e seleção por horizonte.
    """),
    md(r"""
    ## Objetivos e roteiro

    Definir origens temporais; evitar leakage; comparar baseline ingênuo e AR(1); calcular MAE,
    RMSE, cobertura, largura e CRPS; distinguir calibração de sharpness; agregar por horizonte;
    selecionar sem usar o conjunto final de teste.
    """),
    md(r"""
    ## Mapa conceitual: o protocolo faz parte do problema

    Um resultado de forecast só é interpretável quando quatro escolhas estão fixadas:

    | Elemento | Pergunta |
    |---|---|
    | origem $T$ | até qual instante o modelo pode enxergar? |
    | horizonte $h$ | quanto tempo à frente queremos prever? |
    | janela | histórico cresce, desliza ou fica fixo? |
    | regra de atualização | parâmetros são reestimados a cada origem? |

    Alterar qualquer item muda o experimento. Comparar um modelo rolling de um passo com outro
    multi-step de origem fixa não mede apenas modelos: mede protocolos diferentes. “Sem leakage”
    significa que toda transformação, seleção e estimação em uma origem usa somente dados
    disponíveis até aquela origem.
    """),
    md(r"""
    ## Espinha matemática: previsão, perda e avaliação

    Na origem $T$, um modelo emite uma distribuição preditiva

    $$F_{T,h}(y)=P(Y_{T+h}\le y\mid\mathcal F_T),\qquad h=1,\ldots,H.$$

    Para previsões pontuais $\widehat y_{T,h}$, usamos

    $$\operatorname{MAE}_h=\frac1N\sum_{i=1}^{N}|y_{T_i+h}-\widehat y_{T_i,h}|,$$
    $$\operatorname{RMSE}_h=
    \sqrt{\frac1N\sum_{i=1}^{N}(y_{T_i+h}-\widehat y_{T_i,h})^2}.$$

    Para um intervalo $[L_{T,h},U_{T,h}]$, a cobertura empírica é

    $$\widehat C_h=\frac1N\sum_{i=1}^{N}
    \mathbf1\{L_{T_i,h}\le y_{T_i+h}\le U_{T_i,h}\}.$$

    O CRPS avalia a distribuição completa:

    $$\operatorname{CRPS}(F,y)=\int_{-\infty}^{\infty}
    \bigl(F(z)-\mathbf1\{y\le z\}\bigr)^2dz.$$

    Um score $S$ é próprio quando, para a distribuição verdadeira $G$,
    $E_{Y\sim G}S(G,Y)\le E_{Y\sim G}S(F,Y)$ para qualquer relato $F$. Se $F_{T,h}$ é contínua e
    corretamente calibrada, então $U_{T,h}=F_{T,h}(Y_{T+h})\sim U(0,1)$ — fundamento do PIT.
    Como origens sobrepostas geram perdas dependentes, médias de score são válidas como resumo,
    mas erros-padrão ingênuos que tratam todas as linhas como independentes não são.
    """),
    setup_cell("from scipy.stats import norm"),
    md(r"""
    ## 1. Protocolo rolling-origin

    Em cada origem $T_i$, o modelo vê apenas $1{:}T_i$ e prevê $h=1,\ldots,H$. Depois avançamos a
    origem. Isso reproduz a informação real disponível e gera vários erros por horizonte.

    Origens próximas compartilham grande parte do treino e, para $H>1$, blocos de teste podem se
    sobrepor. Portanto, os erros não são independentes. Rolling-origin aumenta a quantidade de
    evidência, mas não transforma cada linha da tabela em uma réplica independente.
    """),
    code("""
    def rolling_origins(n,initial,horizon,step=1):
        '''Cortes em que origin+horizon não ultrapassa n.'''
        return list(range(initial,n-horizon+1,step))

    origins=rolling_origins(420,initial=180,horizon=12,step=6)
    pd.Series({"número de origens":len(origins),"primeira":origins[0],"última":origins[-1]})
    """),
    md(r"""
    ## 2. Série e candidatos

    Usaremos AR(1) estacionário. O baseline ingênuo mantém o último valor e assume incerteza de
    caminhada aleatória. O candidato AR(1) estima reversão à média. O baseline é parte obrigatória
    da avaliação, não um detalhe.
    """),
    code("""
    def simulate_ar1(n,phi=.78,intercept=.3,sigma=.9,seed=None):
        rng=np.random.default_rng(seed);x=np.empty(n);x[0]=intercept/(1-phi)
        for t in range(1,n):x[t]=intercept+phi*x[t-1]+rng.normal(0,sigma)
        return x

    series=simulate_ar1(420,seed=701)
    fig,ax=plt.subplots(figsize=(11,4));ax.plot(series,color=COLORS["blue"])
    for o in origins[::7]:ax.axvline(o,color=COLORS["gray"],alpha=.3)
    ax.set(title="Série e algumas origens de avaliação",xlabel="tempo");plt.show()
    """),
    code("""
    def fit_ar1(x):
        y=x[1:];X=np.column_stack([np.ones(len(x)-1),x[:-1]])
        c,phi=np.linalg.lstsq(X,y,rcond=None)[0]
        residuals=y-X@np.array([c,phi]);sigma=np.sqrt(residuals@residuals/(len(y)-2))
        return c,phi,sigma

    def forecast_ar1(last,h,c,phi,sigma):
        steps=np.arange(1,h+1);mean=phi**steps*last+c*(1-phi**steps)/(1-phi)
        variance=sigma**2*np.cumsum(phi**(2*np.arange(h)))
        return mean,np.sqrt(variance)

    def forecast_naive(history,h):
        sigma=np.std(np.diff(history),ddof=1);steps=np.arange(1,h+1)
        return np.repeat(history[-1],h),sigma*np.sqrt(steps)
    """),
    md(r"""
    ### 2.1 A distribuição como contrato de inferência

    `forecast_ar1` generaliza a recorrência estimada para todos os horizontes. A média combina a
    memória $\widehat\phi^hX_T$ com a soma geométrica dos interceptos; a variância soma inovações
    futuras independentes com pesos ao quadrado. O par `(mean, std)` é suficiente para descrever a
    normal plug-in e permite que o backtester trate modelos diferentes de maneira uniforme.

    Neste capítulo, a pergunta muda de “como derivar a fórmula?” para “essa distribuição inferida é
    calibrada em novas origens?”. Cobertura, PIT e CRPS verificam empiricamente a generalização. Uma
    faixa bem derivada pode falhar fora da amostra por parâmetros instáveis, distribuição incorreta
    ou mudança de regime.
    """),
    md(r"""
    ## 3. Tabela de previsões prequential

    Guardaremos uma linha por modelo, origem e horizonte. O valor realizado entra na tabela apenas
    depois que a previsão foi produzida. Essa tabela longa é o contrato central do backtester.
    """),
    code("""
    def run_backtest(values,initial,horizon,step=1):
        records=[]
        for origin in rolling_origins(len(values),initial,horizon,step):
            history=values[:origin];actual=values[origin:origin+horizon]
            c,phi,sigma=fit_ar1(history)
            candidates={"AR(1)":forecast_ar1(history[-1],horizon,c,phi,sigma),
                        "Ingênuo":forecast_naive(history,horizon)}
            for model,(mean,std) in candidates.items():
                for h in range(1,horizon+1):
                    records.append({"modelo":model,"origem":origin,"h":h,"realizado":actual[h-1],
                        "média":mean[h-1],"desvio":std[h-1]})
        return pd.DataFrame(records)

    predictions=run_backtest(series,180,12,step=3)
    predictions.head()
    """),
    md(r"""
    ## 4. Métricas pontuais e probabilísticas

    MAE/RMSE avaliam o centro. Cobertura mede calibração do intervalo; largura mede sharpness.
    Um intervalo infinito cobre tudo e é inútil. CRPS é uma proper scoring rule: compara a
    distribuição inteira e recompensa simultaneamente calibração e concentração.

    Uma *proper scoring rule* tem a propriedade de que, em expectativa, o forecaster minimiza o
    score reportando sua distribuição verdadeira. Isso desencoraja intervalos artificialmente
    largos ou confiança excessiva. RMSE é sensível a erros grandes; MAE mede escala típica; CRPS
    generaliza erro absoluto de um ponto para uma distribuição inteira.
    """),
    code("""
    def gaussian_crps(y,mean,std):
        z=(y-mean)/std
        return std*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/np.sqrt(np.pi))

    def add_scores(frame,confidence=.95):
        out=frame.copy();z=norm.ppf((1+confidence)/2)
        out["erro"]=out["realizado"]-out["média"]
        out["lower"]=out["média"]-z*out["desvio"];out["upper"]=out["média"]+z*out["desvio"]
        out["covered"]=(out["realizado"]>=out["lower"])&(out["realizado"]<=out["upper"])
        out["width"]=out["upper"]-out["lower"]
        out["crps"]=gaussian_crps(out["realizado"],out["média"],out["desvio"])
        return out

    scored=add_scores(predictions)
    overall=scored.groupby("modelo").agg(MAE=("erro",lambda x:np.mean(np.abs(x))),
        RMSE=("erro",lambda x:np.sqrt(np.mean(x**2))),cobertura=("covered","mean"),
        largura=("width","mean"),CRPS=("crps","mean"))
    overall.round(4)
    """),
    md(r"""
    ### 4.1 Métricas por horizonte

    Agregar tudo pode esconder degradação. Um modelo pode vencer em $h=1$ e perder em $h=12$.
    A unidade de decisão deve corresponder ao uso real.
    """),
    code("""
    by_horizon=scored.groupby(["modelo","h"]).agg(RMSE=("erro",lambda x:np.sqrt(np.mean(x**2))),
        cobertura=("covered","mean"),largura=("width","mean"),CRPS=("crps","mean")).reset_index()
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for model,group in by_horizon.groupby("modelo"):
        axes[0].plot(group.h,group.RMSE,marker="o",label=model)
        axes[1].plot(group.h,group.cobertura,marker="o",label=model)
        axes[2].plot(group.h,group.CRPS,marker="o",label=model)
    axes[0].set_title("RMSE");axes[1].set_title("Cobertura 95%");axes[1].axhline(.95,color="black",ls="--")
    axes[2].set_title("CRPS");
    for ax in axes:ax.set_xlabel("horizonte h");ax.legend()
    plt.tight_layout();plt.show()
    """),
    md(r"""
    ## 5. Calibração condicional e PIT

    Para forecasts gaussianos, $u=\Phi((y-\mu)/\sigma)$ deve ser aproximadamente uniforme se a
    distribuição estiver calibrada. Formato em U sugere dispersão estreita; formato central,
    dispersão larga; assimetria sugere viés.

    Calibração é uma propriedade de repetição: entre previsões que atribuem 20% a um evento, ele
    deveria ocorrer perto de 20% das vezes. Sharpness é concentração e só é desejável **sujeita à
    calibração**. O PIT avalia mais que um único intervalo de 95%, pois examina toda a CDF prevista.
    """),
    code("""
    scored["PIT"]=norm.cdf((scored["realizado"]-scored["média"])/scored["desvio"])
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(model,group) in zip(axes,scored.groupby("modelo")):
        ax.hist(group.PIT,bins=10,density=True,color=COLORS["blue"],alpha=.7)
        ax.axhline(1,color=COLORS["orange"],ls="--");ax.set(title=f"PIT — {model}",xlabel="quantil preditivo")
    plt.tight_layout();plt.show()
    """),
    md(r"""
    ## 6. Seleção sem contaminar o teste final

    Use um período de validação para escolher modelo/hiperparâmetros e preserve um teste final para
    estimar desempenho após a escolha. Reutilizar o teste para decisões repetidas o transforma em
    treino indireto. Aqui fazemos uma divisão das origens: 70% seleção, 30% auditoria.

    Pense em três camadas temporais: treino interno ajusta parâmetros; validação escolhe família e
    hiperparâmetros; teste final estima desempenho após congelar a decisão. Depois de olhar o teste
    e modificar o modelo, aquele bloco deixou de ser teste final e precisa ser substituído por nova
    evidência futura.
    """),
    code("""
    unique_origins=np.sort(scored.origem.unique());cut=int(.7*len(unique_origins))
    validation_origins=set(unique_origins[:cut]);test_origins=set(unique_origins[cut:])
    validation=scored[scored.origem.isin(validation_origins)]
    final_test=scored[scored.origem.isin(test_origins)]
    val_crps=validation.groupby("modelo").crps.mean().sort_values();selected=val_crps.index[0]
    audit=final_test.groupby("modelo").agg(CRPS=("crps","mean"),cobertura=("covered","mean"),largura=("width","mean"))
    print("Modelo selecionado somente na validação:",selected)
    audit.round(4)
    """),
    md(r"""
    ## 7. Incerteza das métricas

    Previsões de origens próximas se sobrepõem e seus erros são dependentes. Diferenças pequenas
    entre médias não garantem superioridade. Reporte distribuição por origem, estabilidade por
    horizonte e, em trabalhos formais, use inferência que respeite dependência temporal.
    """),
    code("""
    origin_crps=scored.groupby(["modelo","origem"]).crps.mean().unstack(0)
    difference=origin_crps["AR(1)"]-origin_crps["Ingênuo"]
    pd.Series({"diferença média AR1-ingênuo":difference.mean(),"mediana":difference.median(),
        "fração de origens vencidas pelo AR1":np.mean(difference<0),"desvio entre origens":difference.std()}).round(4)
    """),
    md(r"""
    ## 8. Exercícios

    1. Desenhe origens para `n=20`, inicial 10, horizonte 3. 2. Explique calibração versus sharpness.
    3. Por que CRPS é preferível a escolher só cobertura? 4. Refaça com horizonte 24.
    5. Dê um exemplo de leakage em feature temporal. 6. Por que o teste final deve ser usado uma vez?
    """),
    md(r"""
    <details><summary><strong>Gabarito comentado</strong></summary>

    1. Origens: 10, 12, 14 e 16 com `step=2`. 2. Calibração mede frequência; sharpness, concentração.
    3. CRPS penaliza simultaneamente erro e dispersão; cobertura isolada premia intervalos enormes.
    4. Horizontes longos normalmente ampliam incerteza. 5. Média móvel calculada incluindo o alvo.
    6. Decisões repetidas adaptam-se ao teste e tornam sua estimativa otimista.
    </details>
    """),
    code("""
    assert rolling_origins(20,10,3,2)==[10,12,14,16]
    assert len(predictions)==len(rolling_origins(420,180,12,3))*2*12
    assert np.all(scored.desvio>0) and np.all(scored.crps>=0)
    assert selected in {"AR(1)","Ingênuo"}
    print("✓ Gabarito estrutural e testes rápidos passaram.")
    """),
    md(r"""
    ## Síntese

    Um backtest é um experimento temporal versionável: define origem, horizonte, janela, informação,
    modelos, métricas e regra de seleção antes de olhar resultados. Métrica pontual, calibração e
    sharpness respondem perguntas diferentes. O Notebook 08 reunirá esses contratos em um sistema
    end-to-end com artefatos e monitoramento prequential.
    """),
]

if __name__ == "__main__":
    write_notebook(PATH,CELLS,7)

