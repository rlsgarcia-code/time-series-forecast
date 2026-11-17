from pathlib import Path

from notebook_builder import code, md, setup_cell, write_notebook


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "13_sistema_final.ipynb"


CELLS = [
    md(r"""
    # Notebook 13 — Sistema integrado de forecasting, monitoramento e alertas

    **Da série bruta à decisão auditável, preservando a ordem temporal**

    Este capítulo fecha a trilha conectando contrato de dados, modelos candidatos, backtesting,
    seleção, forecast probabilístico imutável, avaliação prequential, CUSUM/EWMA, confirmação
    offline de mudança, risco futuro e política de alertas.
    """),
    md(r"""
    ## Objetivos e critério de conclusão

    Ao final, você executará o fluxo

    `dados → contrato → candidatos → backtest → seleção → forecast congelado → realizado → monitor → alerta → auditoria`.

    O sistema é considerado correto quando:

    1. nenhuma etapa usa informação posterior ao cutoff declarado;
    2. cada realizado é comparado com uma previsão produzida antes dele;
    3. artefatos antigos não podem ser sobrescritos silenciosamente;
    4. seleção e avaliação final usam períodos distintos;
    5. anomalia, drift detectado e risco futuro permanecem conceitos separados;
    6. um alerta guarda previsão, evidência, política e instante que o produziram.
    """),
    md(r"""
    ## 1. Espinha matemática do fluxo

    Na origem $T$, um modelo $m$ transforma a informação legal $\mathcal F_T$ em distribuições
    futuras:

    $$\mathcal M_m(\mathcal F_T)=\{P_{T,h}^{(m)}:h=1,\ldots,H\}.$$

    O backtest estima o risco prequential em origens anteriores $\mathcal I$:

    $$\widehat R(m)=\frac{1}{|\mathcal I|H}
    \sum_{T\in\mathcal I}\sum_{h=1}^{H}
    L(P_{T,h}^{(m)},Y_{T+h}).$$

    Depois de selecionar e congelar a regra, cada previsão de um passo produz

    $$Z_t=\frac{Y_t-\mu_{t\mid t-1}}{\sigma_{t\mid t-1}},$$

    que alimenta detectores causais. Em paralelo, risco futuro responde perguntas como

    $$P(Y_{T+h}>L\mid\mathcal F_T)$$

    ou $P(\max_{1\le h\le H}Y_{T+h}>L\mid\mathcal F_T)$. Evidência de mudança já iniciada não é a
    mesma quantidade que probabilidade de um evento ainda futuro.
    """),
    setup_cell("from dataclasses import asdict, dataclass\nfrom pathlib import Path\nimport tempfile\nfrom scipy.stats import norm\nfrom examples.memory_repository import MemoryRepository\nfrom time_series_lab import AR1Spec, ForecastEngine, MonitoringPolicy, Observation, ThresholdRisk, TimeSeries\nfrom time_series_lab.backtesting import gaussian_crps\nfrom time_series_lab.change_points import optimal_mean_segmentation\nfrom time_series_lab.operations import AlertPolicy, ForecastEnvelope, LocalArtifactStore, SequentialMonitor, gaussian_exceedance_probability, path_exceedance_probability\nfrom time_series_lab.state_space import kalman_filter_local_level, forecast_local_level"),
    md(r"""
    ## 2. Contrato temporal

    Valores sem timestamps não permitem verificar ordem, duplicação ou frequência. O contrato
    abaixo exige índice crescente, único e diário, além de valores finitos e histórico mínimo. Em
    produção também registraríamos timezone, disponibilidade e revisões da fonte.

    O cutoff é a última posição que pode entrar em ajuste, transformação e seleção. “Dados existem
    no banco” não implica “dados estavam disponíveis no instante da decisão”.
    """),
    code("""
    @dataclass(frozen=True)
    class TemporalSeries:
        timestamps: pd.DatetimeIndex
        values: np.ndarray
        frequency: str

        def validate(self,minimum_history=100):
            index=pd.DatetimeIndex(self.timestamps);values=np.asarray(self.values,dtype=float)
            if len(index)!=len(values) or len(values)<minimum_history:raise ValueError("tamanho inválido")
            if not index.is_monotonic_increasing or not index.is_unique:raise ValueError("timestamps fora de ordem")
            if not np.all(np.isfinite(values)):raise ValueError("valores não finitos")
            expected=pd.date_range(index[0],periods=len(index),freq=self.frequency)
            if not index.equals(expected):raise ValueError("frequência ou gaps incompatíveis")
            return values

    def simulate_operational_series(n=460,change_at=350,seed=1301):
        rng=np.random.default_rng(seed);x=np.empty(n);x[0]=0
        for t in range(1,n):
            shift=0 if t<change_at else .75
            x[t]=.15+shift+.78*x[t-1]+rng.normal(0,.8)
        return x

    timestamps=pd.date_range("2025-01-01",periods=460,freq="D",tz="UTC")
    values=TemporalSeries(timestamps,simulate_operational_series(),"D").validate()
    CHANGE_AT=350
    fig,ax=plt.subplots(figsize=(12,4));ax.plot(timestamps,values,color=COLORS["blue"])
    ax.axvline(timestamps[CHANGE_AT],color=COLORS["red"],ls="--",label="mudança simulada")
    ax.legend();ax.set_title("Série operacional com mudança posterior à seleção");plt.show()
    """),
    md(r"""
    ## 3. Interface probabilística de modelos

    Todos os candidatos obedecem à assinatura

    `history, horizon -> mean, std, parameters`.

    O AR(1) usa OLS e a distribuição multi-step já derivada. O nível local estima variâncias por
    momentos das diferenças: para $\Delta Y_t=\eta_t+\varepsilon_t-\varepsilon_{t-1}$,

    $$\gamma_{\Delta}(0)=q+2r,\qquad \gamma_{\Delta}(1)=-r.$$

    Logo, estimamos $\widehat r=-\widehat\gamma_{\Delta}(1)$ e
    $\widehat q=\widehat\gamma_{\Delta}(0)-2\widehat r$, truncando apenas para preservar positividade.
    Isso é simples e rápido para o laboratório; máxima verossimilhança seria preferível no artefato
    final de maior criticidade.
    """),
    code("""
    def predict_ar1(history,horizon):
        x=np.asarray(history);y=x[1:];X=np.column_stack([np.ones(len(x)-1),x[:-1]])
        c,phi=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@np.array([c,phi])
        sigma=np.sqrt(res@res/(len(y)-2));h=np.arange(1,horizon+1)
        if abs(phi-1)<1e-8:
            mean=x[-1]+h*c;variance=sigma**2*h
        else:
            mean=phi**h*x[-1]+c*(1-phi**h)/(1-phi)
            variance=sigma**2*np.cumsum(phi**(2*np.arange(horizon)))
        return mean,np.sqrt(variance),{"c":c,"phi":phi,"sigma":sigma}

    def predict_local_level(history,horizon):
        x=np.asarray(history);d=np.diff(x);gamma0=np.var(d,ddof=1)
        gamma1=np.mean((d[1:]-d[1:].mean())*(d[:-1]-d[:-1].mean()))
        r=max(-gamma1,1e-4);q=max(gamma0-2*r,1e-4)
        filtered=kalman_filter_local_level(x,initial_mean=x[0],initial_variance=10,
            state_variance=q,observation_variance=r)
        forecast=forecast_local_level(filtered,horizon=horizon,state_variance=q,observation_variance=r)
        return forecast.mean,np.sqrt(forecast.observation_variance),{"q":q,"r":r}

    MODEL_REGISTRY={"ar1":predict_ar1,"local_level":predict_local_level}
    """),
    md(r"""
    ## 4. Seleção somente antes da mudança

    Usaremos origens entre 180 e `SELECTION_END=330`, horizonte 6 e passo 5. Para cada origem,
    todos os modelos veem o mesmo histórico e são avaliados nos mesmos realizados. CRPS é a métrica
    principal; cobertura funciona como guardrail.

    O período posterior a 330 não participa da escolha. Se mudássemos candidatos ou parâmetros após
    ver os resultados de monitoramento, precisaríamos registrar uma nova rodada de pesquisa.
    """),
    code("""
    def backtest_registry(values,models,*,initial,end,horizon,step):
        rows=[]
        for origin in range(initial,end-horizon+1,step):
            history=values[:origin];actual=values[origin:origin+horizon]
            for name,predict in models.items():
                mean,std,_=predict(history,horizon)
                for h in range(horizon):rows.append({"model":name,"origin":origin,"h":h+1,
                    "actual":actual[h],"mean":mean[h],"std":std[h]})
        return pd.DataFrame(rows)

    SELECTION_END=330
    bt=backtest_registry(values,MODEL_REGISTRY,initial=180,end=SELECTION_END,horizon=6,step=5)
    bt["crps"]=gaussian_crps(bt.actual,bt["mean"],bt["std"])
    z95=norm.ppf(.975);bt["covered"]=(bt.actual>=bt["mean"]-z95*bt["std"])&(
        bt.actual<=bt["mean"]+z95*bt["std"])
    scoreboard=bt.groupby("model").agg(crps=("crps","mean"),coverage=("covered","mean"),
        width=("std",lambda s:2*z95*s.mean())).sort_values("crps")
    scoreboard.round(4)
    """),
    code("""
    eligible=scoreboard[(scoreboard.coverage>=.88)&(scoreboard.coverage<=.99)]
    if eligible.empty:raise RuntimeError("nenhum modelo passou o guardrail de cobertura")
    SELECTED_MODEL=eligible.index[0]
    SELECTION_RECORD={"metric":"crps","guardrail_coverage":[.88,.99],
        "selected":SELECTED_MODEL,"selection_end":SELECTION_END,
        "scores":scoreboard.reset_index().to_dict(orient="records")}
    pd.Series(SELECTION_RECORD)
    """),
    md(r"""
    ## 5. Artefato imutável e identificador por conteúdo

    Cada forecast contém modelo, cutoff, horizontes, média, desvio e metadados. Um hash SHA-256 do
    conteúdo gera `forecast_id`. O armazenamento abre o arquivo em modo exclusivo: tentar salvar o
    mesmo identificador novamente falha em vez de sobrescrever evidência.

    O hash não substitui controle de acesso ou assinatura criptográfica, mas detecta alterações
    acidentais e liga monitoramento à distribuição efetivamente emitida.
    """),
    code("""
    artifact_directory=Path(tempfile.mkdtemp(prefix="time-series-artifacts-"))
    store=LocalArtifactStore(artifact_directory)

    mean,std,params=MODEL_REGISTRY[SELECTED_MODEL](values[:SELECTION_END],12)
    initial_artifact=ForecastEnvelope.create(model_name=SELECTED_MODEL,cutoff=SELECTION_END-1,
        horizons=np.arange(1,13),mean=mean,std=std,
        metadata={"parameters":params,"selection":SELECTION_RECORD,"frequency":"D","timezone":"UTC"})
    initial_path=store.save(initial_artifact)
    restored=store.load(initial_artifact.forecast_id)
    pd.Series({"forecast_id":restored.forecast_id,"cutoff":restored.cutoff,
        "modelo":restored.model_name,"arquivo":str(initial_path)})
    """),
    md(r"""
    ## 6. Monitor prequential com política congelada

    Para cada $t\ge330$:

    1. ajustamos o modelo apenas em `values[:t]`;
    2. emitimos e persistimos a distribuição de $Y_t$ com cutoff $t-1$;
    3. somente então revelamos `values[t]`;
    4. calculamos $Z_t$ e atualizamos CUSUM/EWMA;
    5. a política aplica cooldown e reset definidos antes do fluxo.

    O evento guarda `forecast_id`, permitindo reconstruir exatamente a expectativa que foi violada.
    """),
    code("""
    POLICY=AlertPolicy(point_threshold=3.0,cusum_reference=.4,cusum_threshold=6.0,
        ewma_smoothing=.15,ewma_limit_multiplier=3.6,cooldown=14,reset_on_alert=True)
    monitor=SequentialMonitor(POLICY);monitor_rows=[]
    for t in range(SELECTION_END,len(values)):
        mean,std,params=MODEL_REGISTRY[SELECTED_MODEL](values[:t],1)
        artifact=ForecastEnvelope.create(model_name=SELECTED_MODEL,cutoff=t-1,horizons=[1],
            mean=mean,std=std,metadata={"parameters":params,"target_timestamp":str(timestamps[t])})
        store.save(artifact)
        event=monitor.update(forecast_id=artifact.forecast_id,actual=values[t],mean=mean[0],std=std[0])
        monitor_rows.append({"t":t,"timestamp":timestamps[t],**asdict(event)})
    monitoring=pd.DataFrame(monitor_rows)
    monitoring.loc[monitoring.emitted,["t","timestamp","surprise","reasons","forecast_id"]].head(10)
    """),
    code("""
    fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True)
    axes[0].plot(monitoring.t,monitoring.actual,color="black");axes[0].axvline(CHANGE_AT,color=COLORS["red"],ls="--")
    axes[0].set_title("Realizado no período de monitoramento")
    axes[1].plot(monitoring.t,monitoring.surprise,color=COLORS["blue"],alpha=.6);axes[1].axhline(0,color="black")
    axes[1].set_ylabel("surpresa")
    axes[2].plot(monitoring.t,monitoring.cusum_positive,color=COLORS["green"],label="CUSUM+")
    axes[2].plot(monitoring.t,monitoring.ewma,color=COLORS["purple"],label="EWMA")
    alerts=monitoring[monitoring.emitted]
    axes[2].scatter(alerts.t,alerts.ewma,color=COLORS["red"],zorder=5,label="alerta emitido")
    axes[2].axvline(CHANGE_AT,color=COLORS["red"],ls="--");axes[2].legend();axes[2].set_xlabel("tempo")
    plt.tight_layout();plt.show()
    """),
    md(r"""
    ## 7. Separar anomalia, drift e confirmação retrospectiva

    - `point_anomaly` indica uma realização individual improvável.
    - CUSUM/EWMA indicam evidência sequencial acumulada.
    - change-point offline usa toda a janela posterior para estimar segmentos retrospectivamente.

    A segmentação não deve alterar o instante original do alerta. Ela serve para investigação,
    estimativa de localização e possível redesenho do modelo.
    """),
    code("""
    segmentation=optimal_mean_segmentation(monitoring.surprise.to_numpy(),penalty=14,min_size=15)
    retrospective_breaks=tuple(SELECTION_END+point for point in segmentation.breakpoints)
    first_emitted=int(alerts.t.iloc[0]) if len(alerts) else None
    pd.Series({"mudança verdadeira":CHANGE_AT,"primeiro alerta emitido":first_emitted,
        "quebras retrospectivas":retrospective_breaks,
        "atraso do primeiro alerta":None if first_emitted is None else first_emitted-CHANGE_AT})
    """),
    md(r"""
    ## 8. Risco marginal e risco de trajetória

    No cutoff inicial, a probabilidade marginal gaussiana é

    $$P(Y_{T+h}>L)=1-\Phi\left(\frac{L-\mu_h}{\sigma_h}\right).$$

    Já o evento $\max_hY_{T+h}>L$ depende da distribuição conjunta. Para AR(1), simularemos
    caminhos recursivos compartilhando corretamente os choques ao longo dos horizontes:

    $$\widehat P(\max_hY_{T+h}>L)=\frac1B\sum_{b=1}^{B}
    \mathbf1\{\max_hY_{T+h}^{(b)}>L\}.$$

    Somar probabilidades marginais contaria trajetórias repetidas e pode exceder 1.
    """),
    code("""
    RISK_THRESHOLD=5.0
    marginal_risk=gaussian_exceedance_probability(initial_artifact.mean,initial_artifact.std,RISK_THRESHOLD)

    def simulate_ar1_paths(last,horizon,n_paths,params,seed=0):
        rng=np.random.default_rng(seed);paths=np.empty((n_paths,horizon));current=np.repeat(last,n_paths)
        for h in range(horizon):
            current=params["c"]+params["phi"]*current+rng.normal(0,params["sigma"],n_paths)
            paths[:,h]=current
        return paths

    if SELECTED_MODEL=="ar1":
        risk_paths=simulate_ar1_paths(values[SELECTION_END-1],12,20_000,params=initial_artifact.metadata["parameters"],seed=1302)
    else:
        rng=np.random.default_rng(1302);q=initial_artifact.metadata["parameters"]["q"];r=initial_artifact.metadata["parameters"]["r"]
        latent=np.cumsum(rng.normal(0,np.sqrt(q),size=(20_000,12)),axis=1)+initial_artifact.mean[0]
        risk_paths=latent+rng.normal(0,np.sqrt(r),size=latent.shape)
    joint_risk=path_exceedance_probability(risk_paths,RISK_THRESHOLD)
    pd.Series({"maior risco marginal":marginal_risk.max(),"risco em qualquer horizonte":joint_risk,
        "soma incorreta das marginais":marginal_risk.sum()}).round(4)
    """),
    md(r"""
    ## 9. Auditoria e invariantes

    Uma auditoria reproduzível verifica relações, não apenas arquivos existentes:

    - `artifact.cutoff < target_time`;
    - `forecast_id` corresponde ao conteúdo armazenado;
    - cada evento referencia artefato recuperável;
    - seleção usa origens anteriores ao período monitorado;
    - cooldown separa alertas emitidos;
    - risco marginal e de trajetória permanecem identificados.
    """),
    code("""
    audit={
        "artefatos persistidos":len(list(artifact_directory.glob("*.json"))),
        "eventos monitorados":len(monitoring),
        "alertas brutos":int(monitoring.reasons.map(bool).sum()),
        "alertas emitidos":int(monitoring.emitted.sum()),
        "primeiro alerta":first_emitted,
        "modelo selecionado":SELECTED_MODEL,
        "CRPS de seleção":float(scoreboard.loc[SELECTED_MODEL,"crps"]),
        "risco conjunto":joint_risk,
    }
    pd.Series(audit)
    """),
    md(r"""
    ## 10. O que cada capítulo acrescentou ao sistema

    | Capítulos | Responsabilidade incorporada |
    |---|---|
    | 01–04 | processos, estimação, ARMA/ARIMA e forecast probabilístico |
    | 05–06 | simulação contínua, Itô e volatilidade condicional |
    | 07–08 | backtest, seleção, contratos e primeiro protótipo integrado |
    | 09 | estados latentes e filtro de Kalman |
    | 10 | regimes discretos e probabilidades de mudança de regime |
    | 11 | confirmação retrospectiva de quebras estruturais |
    | 12 | detecção causal com falso alarme e atraso calibrados |
    | 13 | persistência, proveniência, política e auditoria end-to-end |

    O registry é extensível: Kalman e Markov switching podem ser promovidos a candidatos desde que
    implementem o mesmo contrato de distribuição e sejam avaliados no mesmo protocolo.
    """),
    md(r"""
    ## 11. A mesma operação pela API Python

    As células anteriores mantiveram cada fórmula e transição explícita. Para uma aplicação, porém,
    não queremos copiar esse código de orquestração. `ForecastEngine` oferece a fachada estável e
    recebe um `EngineRepository` implementado pela aplicação. O repositório em memória abaixo é
    apenas didático; ele torna o exemplo executável sem impor banco à biblioteca.

    O forecast continua sendo emitido **antes** da observação. `process_observation` recupera esse
    artefato, calcula a surpresa pela CDF preditiva, atualiza CUSUM/EWMA, anexa o realizado e emite a
    distribuição do ciclo seguinte.
    """),
    code("""
    api_repository=MemoryRepository()
    api_engine=ForecastEngine(repository=api_repository)
    api_series=TimeSeries("serie-operacional",timestamps[:SELECTION_END],values[:SELECTION_END],"D")
    api_engine.register_series(api_series)

    api_run=api_engine.fit("serie-operacional",AR1Spec())
    api_engine.promote("serie-operacional",api_run.model_run_id)
    api_forecast=api_engine.forecast("serie-operacional",horizon=12,seed=1302)
    api_simulation=api_engine.simulate("serie-operacional",horizon=12,n_paths=5_000,seed=1302)
    api_risk=api_engine.assess_risk(api_simulation,ThresholdRisk(limit=RISK_THRESHOLD))

    pd.Series({
        "modelo":api_run.spec.family,
        "forecast_id":api_forecast.forecast_id,
        "cutoff":api_forecast.cutoff,
        "risco em qualquer horizonte":api_risk.any_horizon_probability,
        "trajetórias":api_simulation.paths.shape[0],
    })
    """),
    code("""
    api_session=api_engine.start_monitoring(
        "serie-operacional",
        MonitoringPolicy(refit_each_observation=True,forecast_horizon=1),
    )
    api_cycle=api_engine.process_observation(
        api_session.session_id,
        Observation(timestamps[SELECTION_END],values[SELECTION_END],"notebook-13-observation-330"),
    )
    pd.Series({
        "forecast avaliado":api_cycle.monitoring_event.forecast_id,
        "surpresa":api_cycle.monitoring_event.surprise,
        "razões":api_cycle.monitoring_event.reasons,
        "próximo cutoff":api_cycle.next_forecast.cutoff,
        "próximo alvo":api_cycle.next_forecast.targets[0],
    })
    """),
    md(r"""
    ## 12. Limites do protótipo

    O fluxo é operacionalmente coerente, mas ainda é local e univariado. Uma implantação real
    acrescentaria banco transacional ou object store, controle de concorrência, autenticação,
    agendamento, observabilidade, schema versionado, reconciliação de dados tardios e testes de
    recuperação.

    Estatisticamente, também seriam necessários intervalos que propaguem incerteza de parâmetros,
    famílias não gaussianas quando apropriado, calibração por população e tratamento de múltiplas
    séries/alertas. Essas limitações não invalidam o contrato causal demonstrado; delimitam a
    passagem de protótipo reproduzível para serviço de produção.
    """),
    md(r"""
    ## 13. Exercícios

    1. Adicione `local_level` ao resultado final mesmo que não seja selecionado e compare risco.
    2. Faça o guardrail de cobertura variar por horizonte.
    3. Provoque adulteração de um JSON copiado e confirme que `load` rejeita seu hash.
    4. Mude `cooldown` e `reset_on_alert`; compare alertas brutos e emitidos.
    5. Acrescente um detector de mudança de variância sobre $Z_t^2$.
    6. Registre um evento de qualidade de dados separado de anomalia estatística.
    7. Promova Markov switching ao registry usando média e variância da mistura.
    8. Desenhe uma política explícita de retreino que exija confirmação e conjunto de validação.
    """),
    md(r"""
    <details><summary><strong>Gabarito orientativo</strong></summary>

    1. Execute ambos no mesmo cutoff e preserve artefatos distintos.
    2. Agrupe o backtest por `h` e rejeite modelos fora da faixa nos horizontes críticos.
    3. O conteúdo alterado gera digest diferente do `forecast_id` preservado.
    4. Cooldown maior reduz notificações repetidas; reset muda a memória pós-alerta.
    5. Padronize $Z_t^2-1$ ou use modelo específico de volatilidade e calibre novamente.
    6. Gap, atraso ou unidade incorreta devem interromper o pipeline antes do detector.
    7. Retorne momentos da mistura e, idealmente, amostras/quantis para preservar multimodalidade.
    8. Separe alerta, investigação, candidato retreinado, backtest e promoção auditada.

    </details>
    """),
    code("""
    assert restored==initial_artifact and restored.verify()
    assert SELECTED_MODEL in MODEL_REGISTRY
    assert bt.origin.max()+6<=SELECTION_END
    assert np.all(monitoring.t>monitoring.sequence+SELECTION_END-2)
    assert all(store.load(fid).cutoff < t for fid,t in zip(monitoring.forecast_id,monitoring.t))
    emitted_sequences=monitoring.loc[monitoring.emitted,"sequence"].to_numpy()
    assert len(emitted_sequences)<2 or np.all(np.diff(emitted_sequences)>POLICY.cooldown)
    assert 0<=joint_risk<=1 and np.all((marginal_risk>=0)&(marginal_risk<=1))
    assert len(list(artifact_directory.glob("*.json")))==len(monitoring)+1
    assert api_cycle.monitoring_event.forecast_id is not None
    assert api_cycle.next_forecast.cutoff==timestamps[SELECTION_END]
    print("Testes rápidos do Notebook 13 passaram.")
    """),
    md(r"""
    ## Conclusão

    A solução agora possui uma cadeia verificável entre matemática, código e operação. Forecast é
    uma distribuição condicionada a um cutoff; avaliação é prequential; seleção ocorre antes do
    monitoramento; detectores acumulam apenas evidência disponível; alertas obedecem política
    congelada; e risco futuro permanece separado de mudanças já observadas.

    O resultado é um protótipo univariado end-to-end reproduzível, não apenas uma coleção de
    notebooks. Seus contratos permitem evoluir armazenamento, modelos e detectores sem perder a
    proveniência das decisões.
    """),
]


if __name__ == "__main__":
    write_notebook(PATH, CELLS, 13)
