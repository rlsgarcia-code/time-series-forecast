"""Reconstrói o Notebook 01 a partir de células versionáveis neste script."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_processos_estocasticos.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


CELLS = [
    md(
        r"""
        # Notebook 01 — Processos estocásticos e previsão probabilística

        **Ruído branco → passeio aleatório → AR(1) → forecast probabilístico**

        Este é o primeiro laboratório do **Time Series Lab**. A ideia é construir intuição
        visual sem abrir mão das fórmulas e de verificações numéricas reproduzíveis.

        > **Como usar:** execute as células em ordem. Quando encontrar um controle deslizante,
        > mova-o e observe o que muda mantendo as mesmas inovações aleatórias.
        """
    ),
    md(
        r"""
        ## Para quem é este notebook

        **Público:** quem já viu probabilidade básica e quer entender os blocos de construção
        de modelos de séries temporais.

        **Pré-requisitos:** média, variância, correlação e leitura de gráficos. Cálculo não é
        necessário; as fórmulas importantes são explicadas no próprio fluxo.

        ### Objetivos de aprendizagem

        Ao final, você será capaz de:

        1. distinguir ruído branco, passeio aleatório e AR(1);
        2. explicar estacionariedade, reversão à média e o papel de $\phi$;
        3. interpretar uma ACF e relacioná-la à persistência;
        4. medir como um choque desaparece (ou alterna de sinal);
        5. construir e interpretar uma previsão **como distribuição**, não só como um número.

        ### Roteiro

        1. Preparação e reprodutibilidade
        2. Ruído branco
        3. Passeio aleatório e 100 trajetórias
        4. AR(1), slider de $\phi$ e ACF
        5. Choques e persistência
        6. Forecast probabilístico analítico + Monte Carlo
        7. Exercícios, gabarito e testes rápidos
        """
    ),
    md(
        r"""
        ## Espinha matemática do capítulo

        Trabalharemos em um espaço de probabilidade $(\Omega,\mathcal F,P)$ e usaremos
        $\mathcal F_t$ para representar toda a informação disponível até o instante $t$.
        Escrever $E_t[Y]$ significa $E[Y\mid\mathcal F_t]$: a melhor previsão quadrática de $Y$
        usando apenas a informação conhecida em $t$.

        Os três processos do notebook podem ser comparados por suas representações fechadas:

        | Processo | Definição | Média | Variância e dependência |
        |---|---|---|---|
        | ruído branco | $\varepsilon_t\overset{iid}{\sim}N(0,\sigma^2)$ | $E[\varepsilon_t]=0$ | $\operatorname{Cov}(\varepsilon_t,\varepsilon_s)=0$ para $t\ne s$ |
        | passeio aleatório | $X_t=X_{t-1}+d+\varepsilon_t$ | $E[X_t]=X_0+dt$ | $\operatorname{Var}(X_t)=t\sigma^2$ e $\operatorname{Cov}(X_s,X_t)=\sigma^2\min(s,t)$ |
        | AR(1) | $X_t=c+\phi X_{t-1}+\varepsilon_t$ | $E[X_t]\to\mu=c/(1-\phi)$ se $|\phi|<1$ | $\gamma_k=\sigma^2\phi^{|k|}/(1-\phi^2)$ em estacionariedade |

        A solução recursiva do AR(1), condicionada a $X_0$, é

        $$X_t=\mu+\phi^t(X_0-\mu)+\sum_{j=0}^{t-1}\phi^j\varepsilon_{t-j}.$$

        Essa identidade contém quase todo o capítulo: o estado inicial perde peso como $\phi^t$;
        cada choque recebe peso $\phi^j$; e a variância soma os quadrados desses pesos. Para
        previsão a partir de $T$,

        $$X_{T+h}\mid\mathcal F_T\sim N\!\left(
        \mu+\phi^h(X_T-\mu),\;\sigma^2\sum_{j=0}^{h-1}\phi^{2j}
        \right).$$

        Mais adiante derivaremos essa distribuição começando por $h=1$ e $h=2$, em vez de apenas
        utilizá-la como fórmula pronta.
        """
    ),
    md(
        r"""
        ## 0. Preparação

        Todas as simulações usam sementes explícitas. Isso significa que, ao executar novamente,
        você verá os mesmos números — essencial para comparar modelos sem confundir mudança de
        parâmetro com uma nova amostra aleatória.
        """
    ),
    code(
        """
        from dataclasses import dataclass
        from importlib.metadata import version

        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import display
        from scipy.stats import norm

        %matplotlib inline

        COLORS = {
            "blue": "#2563EB",
            "orange": "#EA580C",
            "green": "#059669",
            "purple": "#7C3AED",
            "gray": "#64748B",
            "red": "#DC2626",
        }
        plt.rcParams.update({
            "figure.figsize": (10, 4.8),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "axes.titleweight": "bold",
            "font.size": 11,
        })

        pd.DataFrame(
            {"versão": [version(pkg) for pkg in ["numpy", "pandas", "matplotlib", "ipywidgets"]]},
            index=["numpy", "pandas", "matplotlib", "ipywidgets"],
        )
        """
    ),
    md(
        r"""
        ## 1. Ruído branco

        Um ruído branco gaussiano é uma sequência

        $$\varepsilon_t \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2).$$

        Cada valor tem a mesma distribuição e não carrega memória dos valores anteriores.
        Uma realização curta pode parecer ter padrões; isso não implica dependência real.

        A célula abaixo traduz a equação diretamente: o gerador cria valores normais
        independentes com média `mu` e desvio-padrão `sigma`.
        """
    ),
    code(
        """
        def simulate_white_noise(n_steps, *, n_paths=1, mu=0.0, sigma=1.0, seed=None):
            '''Gera ε_t ~ N(mu, sigma²), independente para cada t e trajetória.'''
            rng = np.random.default_rng(seed)

            # Cada posição é uma inovação: ε_t = mu + sigma · Z_t,
            # onde Z_t ~ N(0, 1) e não depende dos demais instantes.
            epsilon = rng.normal(
                loc=mu,
                scale=sigma,
                size=(n_paths, n_steps),
            )
            return epsilon
        """
    ),
    code(
        """
        white_noise = simulate_white_noise(500, mu=0, sigma=1, seed=101)[0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(white_noise, color=COLORS["blue"], linewidth=1)
        axes[0].axhline(0, color="black", linewidth=1)
        axes[0].set(title="Uma realização de ruído branco", xlabel="tempo", ylabel=r"$\\varepsilon_t$")

        axes[1].hist(white_noise, bins=28, density=True, color=COLORS["blue"], alpha=0.75)
        axes[1].set(title="Distribuição empírica", xlabel="valor", ylabel="densidade")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        white_noise_summary = pd.Series(white_noise).agg(["mean", "std", "min", "max"])
        white_noise_summary.rename({"mean": "média", "std": "desvio-padrão", "min": "mínimo", "max": "máximo"}).to_frame("valor")
        """
    ),
    md(
        r"""
        **Leitura:** a média e o desvio-padrão amostrais não são exatamente 0 e 1. Eles se
        aproximam dos parâmetros conforme a amostra cresce. Aleatoriedade não significa
        “sem estrutura visual”; significa que a estrutura probabilística está nas distribuições.
        """
    ),
    md(
        r"""
        ## 2. Passeio aleatório

        O passeio aleatório acumula ruídos:

        $$X_t = X_{t-1} + \varepsilon_t.$$

        A diferença $\Delta X_t = X_t-X_{t-1}$ é ruído branco, mas o **nível** $X_t$ guarda
        para sempre todos os choques passados. Por isso, $\mathrm{Var}(X_t)=t\sigma^2$ cresce
        com o tempo e o processo em nível não é estacionário.

        No código, `np.cumsum` implementa exatamente a soma
        $X_t=X_0+\sum_{j=1}^{t}(d+\varepsilon_j)$.
        """
    ),
    code(
        """
        def simulate_random_walk(
            n_steps, *, n_paths=1, x0=0.0, drift=0.0, sigma=1.0, seed=None
        ):
            '''Simula X_t = X_(t-1) + drift + ε_t.'''
            epsilon = simulate_white_noise(
                n_steps, n_paths=n_paths, sigma=sigma, seed=seed
            )
            increments = drift + epsilon

            # A primeira coluna representa X_0. As demais acumulam todos os choques.
            paths = np.empty((n_paths, n_steps + 1))
            paths[:, 0] = x0
            paths[:, 1:] = x0 + np.cumsum(increments, axis=1)
            return paths
        """
    ),
    code(
        """
        random_walk = simulate_random_walk(250, sigma=1, seed=202)[0]
        increments = np.diff(random_walk)

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(random_walk, color=COLORS["orange"])
        axes[0].set(title="Passeio aleatório: o nível vagueia", ylabel=r"$X_t$")
        axes[1].plot(np.arange(1, len(random_walk)), increments, color=COLORS["gray"], linewidth=1)
        axes[1].set(title="Primeira diferença: recuperamos o ruído", xlabel="tempo", ylabel=r"$\\Delta X_t$")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        r"""
        ### 100 trajetórias: uma observação muda a intuição

        Um único caminho esconde a distribuição do processo. Com 100 trajetórias independentes,
        o “leque” crescente mostra visualmente a variância aumentando com $t$.
        """
    ),
    code(
        """
        rw_100 = simulate_random_walk(250, n_paths=100, sigma=1, seed=203)
        time_rw = np.arange(rw_100.shape[1])

        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(time_rw, rw_100.T, color=COLORS["blue"], alpha=0.12, linewidth=0.9)
        ax.plot(time_rw, rw_100.mean(axis=0), color="black", linewidth=2, label="média entre trajetórias")
        ax.set(title="100 trajetórias de um passeio aleatório", xlabel="tempo", ylabel=r"$X_t$")
        ax.legend()
        plt.show()
        """
    ),
    code(
        """
        rw_many = simulate_random_walk(250, n_paths=8_000, sigma=1, seed=204)
        selected_t = np.array([10, 50, 100, 200, 250])
        empirical_variance = rw_many[:, selected_t].var(axis=0)
        variance_check = pd.DataFrame({
            "t": selected_t,
            "variância empírica": empirical_variance,
            "variância teórica t·σ²": selected_t.astype(float),
            "razão empírica/teórica": empirical_variance / selected_t,
        })
        variance_check.round(3)
        """
    ),
    md(
        r"""
        **Armadilha comum:** uma longa subida em um passeio aleatório não prova uma tendência
        determinística. Choques acumulados conseguem produzir trechos visualmente convincentes.
        """
    ),
    md(
        r"""
        ## 3. AR(1): memória controlada por $\phi$

        O modelo autorregressivo de ordem 1 é

        $$X_t = c + \phi X_{t-1} + \varepsilon_t.$$

        Quando $|\phi|<1$, o processo é estacionário e reverte para
        $\mu=c/(1-\phi)$. Sua variância incondicional é
        $\sigma^2/(1-\phi^2)$.

        - $\phi \approx 0$: pouca memória;
        - $0<\phi<1$: persistência sem memória permanente;
        - $-1<\phi<0$: alternância em torno da média;
        - $\phi=1$: passeio aleatório (raiz unitária);
        - $|\phi|>1$: comportamento explosivo.

        Diferentemente do passeio aleatório, aqui não acumulamos os choques diretamente.
        A cada passo, multiplicamos o estado anterior por $\phi$; por isso a influência de
        $X_{t-h}$ chega ao presente multiplicada por $\phi^h$.
        """
    ),
    code(
        """
        def simulate_ar1(
            n_steps, *, phi, n_paths=1, x0=0.0,
            intercept=0.0, sigma=1.0, seed=None
        ):
            '''Simula recursivamente X_t = c + phi·X_(t-1) + ε_t.'''
            epsilon = simulate_white_noise(
                n_steps, n_paths=n_paths, sigma=sigma, seed=seed
            )
            paths = np.empty((n_paths, n_steps + 1))
            paths[:, 0] = x0

            # A dependência temporal aparece explicitamente nesta recorrência.
            for t in range(1, n_steps + 1):
                paths[:, t] = (
                    intercept
                    + phi * paths[:, t - 1]
                    + epsilon[:, t - 1]
                )
            return paths


        def ar1_stationary_moments(*, phi, intercept=0.0, sigma=1.0):
            '''Média e variância de longo prazo, válidas somente quando |phi| < 1.'''
            if abs(phi) >= 1:
                raise ValueError("Os momentos estacionários exigem |phi| < 1.")

            long_run_mean = intercept / (1 - phi)
            long_run_variance = sigma**2 / (1 - phi**2)
            return long_run_mean, long_run_variance
        """
    ),
    code(
        """
        phi_values = [-0.75, 0.0, 0.55, 0.90]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)

        for ax, phi in zip(axes.flat, phi_values):
            path = simulate_ar1(300, phi=phi, sigma=1, seed=301)[0]
            ax.plot(path, color=COLORS["purple"], linewidth=1)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(rf"$\\phi={phi}$")
            ax.set_xlabel("tempo")
            ax.set_ylabel(r"$X_t$")

        fig.suptitle("Mesmo ruído, diferentes memórias", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        r"""
        ### Slider de $\phi$

        O seed é fixo: ao mover o controle, somente $\phi$ muda. Observe o caminho, a ACF
        amostral e a ACF teórica. O slider fica dentro da região estacionária.
        """
    ),
    code(
        """
        def explore_phi(phi: float = 0.70) -> None:
            series = simulate_ar1(500, phi=phi, sigma=1, seed=302)[0, 1:]
            lags = np.arange(21)
            empirical = sample_acf(series, 20)
            theoretical = theoretical_ar1_acf(phi, 20)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(series, color=COLORS["purple"], linewidth=1)
            axes[0].axhline(0, color="black", linewidth=0.8)
            axes[0].set(title=rf"Caminho AR(1), $\\phi={phi:.2f}$", xlabel="tempo", ylabel=r"$X_t$")

            axes[1].vlines(lags, 0, empirical, color=COLORS["blue"], linewidth=3, label="amostral")
            axes[1].plot(lags, theoretical, "o--", color=COLORS["orange"], label=r"teórica: $\\phi^k$")
            axes[1].axhline(0, color="black", linewidth=0.8)
            axes[1].set(title="Função de autocorrelação", xlabel="lag k", ylabel=r"$\\rho(k)$", ylim=(-1, 1.05))
            axes[1].legend()
            plt.tight_layout()
            plt.show()


        phi_slider = widgets.FloatSlider(
            value=0.70,
            min=-0.95,
            max=0.95,
            step=0.05,
            description="φ",
            continuous_update=False,
            style={"description_width": "30px"},
        )
        widgets.interact(explore_phi, phi=phi_slider)
        """
    ),
    md(
        r"""
        ## 4. ACF: a assinatura temporal

        A função de autocorrelação (ACF) no lag $k$ mede a associação linear entre $X_t$ e
        $X_{t-k}$. Para um AR(1) estacionário:

        $$\rho(k)=\phi^k.$$

        A curva teórica é uma propriedade do modelo; as hastes amostrais oscilam porque foram
        calculadas a partir de uma realização finita.
        """
    ),
    code(
        """
        def theoretical_ar1_acf(phi, nlags):
            '''No AR(1), rho(k) = phi^k para k = 0, ..., nlags.'''
            lags = np.arange(nlags + 1)
            return phi**lags


        def sample_acf(values, nlags):
            '''Correlação amostral entre X_t e X_(t-k), normalizada pelo lag zero.'''
            x = np.asarray(values, dtype=float)
            centered = x - x.mean()
            total_variation = np.sum(centered**2)

            acf = np.empty(nlags + 1)
            acf[0] = 1.0
            for lag in range(1, nlags + 1):
                cross_product = np.sum(centered[lag:] * centered[:-lag])
                acf[lag] = cross_product / total_variation
            return acf
        """
    ),
    code(
        """
        phi_acf = 0.82
        acf_series = simulate_ar1(2_000, phi=phi_acf, sigma=1, seed=401)[0, 200:]
        lags = np.arange(31)
        acf_empirical = sample_acf(acf_series, 30)
        acf_theoretical = theoretical_ar1_acf(phi_acf, 30)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.vlines(lags, 0, acf_empirical, color=COLORS["blue"], linewidth=3, label="ACF amostral")
        ax.plot(lags, acf_theoretical, "o--", color=COLORS["orange"], label=r"ACF teórica $\\phi^k$")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title=r"ACF de um AR(1) com $\\phi=0{,}82$", xlabel="lag k", ylabel="autocorrelação")
        ax.legend()
        plt.show()
        """
    ),
    code(
        """
        acf_comparison = pd.DataFrame({
            "lag": np.arange(6),
            "amostral": acf_empirical[:6],
            "teórica": acf_theoretical[:6],
        })
        acf_comparison.round(3)
        """
    ),
    md(
        r"""
        **Diagnóstico, não veredito:** ACF ajuda a reconhecer dependência, mas não identifica
        sozinha o processo gerador. Tendência, sazonalidade e quebras estruturais também podem
        produzir autocorrelações altas.
        """
    ),
    md(
        r"""
        ## 5. Choques e persistência

        Se dois caminhos de AR(1) recebem exatamente as mesmas inovações, exceto por um choque
        extra de tamanho $\delta$ no instante zero, a diferença futura é

        $$\Delta_h=\delta\phi^h.$$

        Portanto, $\phi$ determina quanto do choque sobrevive. Com $\phi<0$, a resposta alterna
        de sinal; com $\phi$ próximo de 1, o choque desaparece lentamente.
        """
    ),
    code(
        """
        def shock_response(phi, shock, horizon):
            '''Efeito de um choque delta após h passos: delta·phi^h.'''
            h = np.arange(horizon + 1)
            return shock * phi**h
        """
    ),
    code(
        """
        shock = 5.0
        horizon_shock = 30
        shock_phis = [-0.7, 0.4, 0.8, 0.95]

        fig, ax = plt.subplots(figsize=(10, 5))
        for phi, color in zip(shock_phis, [COLORS["purple"], COLORS["green"], COLORS["orange"], COLORS["red"]]):
            response = shock_response(phi, shock, horizon_shock)
            ax.plot(response, marker="o", markersize=3, color=color, label=rf"$\\phi={phi}$")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title="Resposta a um choque isolado", xlabel="horizonte após o choque", ylabel="efeito remanescente")
        ax.legend(ncol=2)
        plt.show()
        """
    ),
    code(
        """
        def half_life(phi: float) -> float:
            # Horizonte contínuo em que resta metade do efeito, para 0 < |phi| < 1.
            return np.log(0.5) / np.log(abs(phi))


        pd.DataFrame({
            "phi": [0.4, 0.8, 0.95, 0.99],
            "meia-vida aproximada": [half_life(phi) for phi in [0.4, 0.8, 0.95, 0.99]],
        }).round(2)
        """
    ),
    md(
        r"""
        **Interpretação:** “estacionário” não é sinônimo de “volta rápido”. Um AR(1) com
        $\phi=0{,}99$ é estacionário em teoria, mas apresenta persistência muito longa em
        amostras finitas.
        """
    ),
    md(
        r"""
        ## 6. Forecast probabilístico

        Até aqui usamos o modelo para gerar séries. Agora fazemos a pergunta inversa:

        > Observamos a série até o instante $T$. O que o modelo diz sobre $X_{T+1}$,
        > $X_{T+2}$, ...?

        Neste capítulo, $c$, $\phi$ e $\sigma$ são tratados como **conhecidos**. O último valor
        $X_T$ também é conhecido. A única fonte de incerteza são as inovações futuras
        $\varepsilon_{T+1},\varepsilon_{T+2},\ldots$, que ainda não observamos.

        Uma previsão probabilística precisa responder a duas perguntas separadas:

        1. **Onde está o centro?** — a média condicional prevista;
        2. **Quão incerto é o futuro?** — a variância ou o intervalo preditivo.
        """
    ),
    md(
        r"""
        ### 6.1 Um passo à frente

        Partimos da própria equação do AR(1):

        $$X_{T+1}=c+\phi X_T+\varepsilon_{T+1}.$$

        No instante $T$, os termos $c$, $\phi$ e $X_T$ são números conhecidos. O único termo
        aleatório é $\varepsilon_{T+1}\sim\mathcal{N}(0,\sigma^2)$. Portanto:

        $$X_{T+1}\mid X_T\sim
        \mathcal{N}\left(c+\phi X_T,\;\sigma^2\right).$$

        A melhor previsão pontual sob erro quadrático é $c+\phi X_T$. O erro-padrão preditivo
        de um passo é $\sigma$ — o tamanho típico de uma nova inovação.
        """
    ),
    code(
        """
        # Exemplo numérico de um passo.
        x_T_demo = 3.0
        phi_demo = 0.8
        intercept_demo = 0.4
        sigma_demo = 1.0

        one_step_mean = intercept_demo + phi_demo * x_T_demo
        one_step_std = sigma_demo

        pd.DataFrame({
            "quantidade": ["último valor X_T", "média prevista", "erro-padrão preditivo"],
            "cálculo": ["3,00", "0,40 + 0,80 × 3,00", "σ = 1,00"],
            "valor": [x_T_demo, one_step_mean, one_step_std],
        })
        """
    ),
    md(
        r"""
        ### 6.2 Dois passos: de onde vem a incerteza acumulada?

        Para prever $T+2$, substituímos a equação de $X_{T+1}$ dentro da equação seguinte:

        $$\begin{aligned}
        X_{T+2}
        &=c+\phi X_{T+1}+\varepsilon_{T+2}\\
        &=c(1+\phi)+\phi^2X_T
          +\phi\varepsilon_{T+1}+\varepsilon_{T+2}.
        \end{aligned}$$

        Agora existem **duas** inovações desconhecidas. A primeira chega a $T+2$ amortecida
        por $\phi$; a segunda chega inteira. Como são independentes, somamos suas variâncias:

        $$\mathbb{E}[X_{T+2}\mid X_T]=c(1+\phi)+\phi^2X_T,$$

        $$\mathrm{Var}(X_{T+2}\mid X_T)
        =\sigma^2\phi^2+\sigma^2=\sigma^2(1+\phi^2).$$

        **Não somamos desvios-padrão.** Somamos variâncias de choques independentes e tiramos
        a raiz quadrada somente no final.
        """
    ),
    md(
        r"""
        ### 6.3 Generalizando para horizonte $h$

        Repetindo a substituição, obtemos:

        $$\mathbb{E}[X_{T+h}\mid X_T]
        =\underbrace{\phi^h X_T}_{\text{memória de }X_T}
        +\underbrace{c\sum_{j=0}^{h-1}\phi^j}_{\text{interceptos acumulados}}.$$

        Para $\phi\neq1$, a soma geométrica resulta em

        $$\mathbb{E}[X_{T+h}\mid X_T]
        =\phi^hX_T+c\frac{1-\phi^h}{1-\phi}.$$

        Se $|\phi|<1$, é mais intuitivo escrever a mesma fórmula usando a média de longo prazo
        $\mu=c/(1-\phi)$:

        $$\mathbb{E}[X_{T+h}\mid X_T]
        =\mu+\phi^h(X_T-\mu).$$

        Ou seja, a distância entre a previsão e $\mu$ encolhe por um fator $\phi$ a cada passo.
        Note que **$c$ não é a média de longo prazo**; $c$ é o intercepto da recorrência.

        A variância preditiva soma o efeito de todas as inovações ainda desconhecidas:

        $$\mathrm{Var}(X_{T+h}\mid X_T)
        =\sigma^2\sum_{j=0}^{h-1}\phi^{2j}
        =\sigma^2\frac{1-\phi^{2h}}{1-\phi^2}.$$
        """
    ),
    code(
        """
        # Decomposição horizonte a horizonte para o mesmo exemplo.
        demo_h = np.arange(1, 9)
        long_run_mean_demo = intercept_demo / (1 - phi_demo)
        memory_component = phi_demo**demo_h * x_T_demo
        intercept_component = (
            intercept_demo * (1 - phi_demo**demo_h) / (1 - phi_demo)
        )
        predictive_variance_demo = (
            sigma_demo**2 * (1 - phi_demo ** (2 * demo_h)) / (1 - phi_demo**2)
        )

        decomposition = pd.DataFrame({
            "h": demo_h,
            "memória de X_T": memory_component,
            "interceptos acumulados": intercept_component,
            "média prevista": memory_component + intercept_component,
            "variância prevista": predictive_variance_demo,
            "desvio-padrão": np.sqrt(predictive_variance_demo),
        })
        decomposition.round(3)
        """
    ),
    code(
        """
        # Centro e incerteza têm movimentos diferentes conforme o horizonte aumenta.
        horizon_demo = np.arange(1, 41)
        mean_demo = (
            long_run_mean_demo
            + phi_demo**horizon_demo * (x_T_demo - long_run_mean_demo)
        )
        variance_demo = (
            sigma_demo**2
            * (1 - phi_demo ** (2 * horizon_demo))
            / (1 - phi_demo**2)
        )
        stationary_variance_demo = sigma_demo**2 / (1 - phi_demo**2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(horizon_demo, mean_demo, color=COLORS["orange"], linewidth=2)
        axes[0].axhline(long_run_mean_demo, color=COLORS["green"], linestyle="--", label=r"média $\\mu$")
        axes[0].set(title="O centro reverte à média", xlabel="horizonte h", ylabel="média prevista")
        axes[0].legend()

        axes[1].plot(horizon_demo, variance_demo, color=COLORS["blue"], linewidth=2)
        axes[1].axhline(stationary_variance_demo, color=COLORS["green"], linestyle="--", label="variância de longo prazo")
        axes[1].set(title="A incerteza cresce até um limite", xlabel="horizonte h", ylabel="variância prevista")
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        r"""
        **Leitura dos dois gráficos:** conforme o horizonte aumenta, esquecemos o valor atual.
        A média prevista se aproxima de $\mu$, enquanto a incerteza aumenta até
        $\sigma^2/(1-\phi^2)$, a variância incondicional do AR(1). Isso é diferente do passeio
        aleatório, cuja incerteza cresce sem limite.

        ### 6.4 Implementação da distribuição analítica

        A função abaixo implementa exatamente as duas fórmulas anteriores. O intervalo central
        de 95% é `média ± 1,96 × desvio-padrão` para uma distribuição normal.
        """
    ),
    code(
        """
        @dataclass(frozen=True)
        class ForecastResult:
            horizon: np.ndarray
            mean: np.ndarray
            std: np.ndarray
            lower: np.ndarray
            upper: np.ndarray
            confidence: float


        def forecast_ar1(
            last_value, horizon, *, phi, intercept=0.0,
            sigma=1.0, confidence=0.95
        ):
            '''Distribuição preditiva analítica de um AR(1) gaussiano.'''
            h = np.arange(1, horizon + 1)

            # E[X_(T+h) | X_T]: memória do último valor + interceptos acumulados.
            if phi == 1:
                predictive_mean = last_value + intercept * h
            else:
                predictive_mean = (
                    phi**h * last_value
                    + intercept * (1 - phi**h) / (1 - phi)
                )

            # Var[X_(T+h) | X_T] = sigma²(1 + phi² + ... + phi^(2(h-1))).
            innovation_weights = phi ** (2 * np.arange(horizon))
            predictive_variance = sigma**2 * np.cumsum(innovation_weights)
            predictive_std = np.sqrt(predictive_variance)

            z = norm.ppf((1 + confidence) / 2)
            return ForecastResult(
                horizon=h,
                mean=predictive_mean,
                std=predictive_std,
                lower=predictive_mean - z * predictive_std,
                upper=predictive_mean + z * predictive_std,
                confidence=confidence,
            )
        """
    ),
    md(
        r"""
        #### 6.4.1 Da fórmula à inferência

        A função não produz apenas uma linha: ela descreve, para cada horizonte $h$, a distribuição
        condicional

        $$X_{T+h}\mid X_T,c,\phi,\sigma
        \sim\mathcal N(m_h,s_h^2).$$

        Cada linha do código corresponde a uma parte dessa afirmação:

        - `h = 1, ..., H` transforma uma fórmula para um horizonte em uma curva de distribuições;
        - `phi**h * last_value` é a memória restante do estado observado $X_T$;
        - `intercept * (1-phi**h)/(1-phi)` é a soma geométrica dos interceptos futuros;
        - `cumsum(phi**(2*j))` acumula as **variâncias** das inovações independentes, ponderadas
          pelo quanto cada choque ainda afeta $T+h$;
        - `sqrt` devolve a incerteza à unidade original de $X$;
        - `norm.ppf((1+confidence)/2)` converte cobertura central em quantil normal bilateral;
        - `mean ± z*std` delimita a distribuição de uma observação futura, não a incerteza da média.

        Neste notebook, $c$, $\phi$ e $\sigma$ são conhecidos. Quando forem estimados, substituí-los
        por $\widehat c$, $\widehat\phi$ e $\widehat\sigma$ produzirá um intervalo **plug-in**. Essa
        generalização é útil, mas ignora a incerteza dos próprios parâmetros; o Notebook 02 torna
        essa distinção explícita.

        A expressão com divisão por $1-\phi$ assume $\phi\neq1$. Para $\phi=1$, o modelo é um
        passeio aleatório com drift: $m_h=X_T+hc$ e $s_h^2=h\sigma^2$.
        """
    ),
    md(
        r"""
        ### 6.5 A mesma previsão por Monte Carlo

        A solução analítica é possível porque o modelo é linear e gaussiano. Monte Carlo chega
        à mesma distribuição por outro caminho:

        1. comece cada trajetória no mesmo $X_T$ observado;
        2. sorteie novas inovações para $T+1,\ldots,T+h$;
        3. repita milhares de vezes;
        4. resuma a distribuição dos valores simulados em cada horizonte.

        A solução analítica é exata e rápida; Monte Carlo é visual e se adapta a modelos nos quais
        a distribuição futura não possui uma fórmula simples.
        """
    ),
    code(
        """
        def simulate_ar1_future_paths(
            last_value, horizon, *, phi, n_paths=1_000,
            intercept=0.0, sigma=1.0, seed=None
        ):
            '''Amostra futuros condicionais começando no último valor observado.'''
            paths_with_origin = simulate_ar1(
                horizon,
                phi=phi,
                n_paths=n_paths,
                x0=last_value,
                intercept=intercept,
                sigma=sigma,
                seed=seed,
            )
            # X_T já foi observado; retornamos somente T+1, ..., T+h.
            return paths_with_origin[:, 1:]
        """
    ),
    md(
        r"""
        ### 6.6 Exemplo completo

        Vamos gerar uma série AR(1), fingir que estamos no último instante observado e prever os
        35 passos seguintes. Usaremos simultaneamente a fórmula analítica e 5.000 futuros
        simulados. Eles devem produzir médias e desvios-padrão muito próximos.
        """
    ),
    code(
        """
        phi, intercept, sigma = 0.82, 0.35, 0.9
        observed = simulate_ar1(
            180, phi=phi, intercept=intercept, sigma=sigma, x0=intercept / (1 - phi), seed=601
        )[0]
        forecast_horizon = 35
        analytic = forecast_ar1(
            observed[-1], forecast_horizon, phi=phi, intercept=intercept, sigma=sigma, confidence=0.95
        )
        future_paths = simulate_ar1_future_paths(
            observed[-1], forecast_horizon, phi=phi, intercept=intercept, sigma=sigma,
            n_paths=5_000, seed=602
        )

        forecast_table = pd.DataFrame({
            "h": analytic.horizon,
            "média": analytic.mean,
            "desvio-padrão": analytic.std,
            "limite 2,5%": analytic.lower,
            "limite 97,5%": analytic.upper,
            "média Monte Carlo": future_paths.mean(axis=0),
        })
        selected_horizons = [0, 1, 2, 4, 9, 19, 34]
        forecast_table.iloc[selected_horizons].round(3)
        """
    ),
    md(
        r"""
        No gráfico abaixo:

        - **preto:** valores já observados;
        - **laranja:** média condicional prevista em cada horizonte;
        - **faixa azul:** intervalo preditivo marginal de 95%;
        - **linhas azuis finas:** 100 futuros completos sorteados dentre os 5.000;
        - **verde tracejado:** média de longo prazo $\mu=c/(1-\phi)$.

        Observe que a linha laranja reverte para a verde e que a faixa se alarga até estabilizar.
        """
    ),
    code(
        """
        observed_time = np.arange(len(observed))
        future_time = observed_time[-1] + analytic.horizon

        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.plot(observed_time[-80:], observed[-80:], color="black", linewidth=2, label="observado")
        ax.plot(future_time, future_paths[:100].T, color=COLORS["blue"], alpha=0.055, linewidth=0.8)
        ax.fill_between(
            future_time, analytic.lower, analytic.upper,
            color=COLORS["blue"], alpha=0.18, label="intervalo preditivo 95%"
        )
        ax.plot(future_time, analytic.mean, color=COLORS["orange"], linewidth=2.5, label="média predita")
        stationary_mean, stationary_variance = ar1_stationary_moments(
            phi=phi, intercept=intercept, sigma=sigma
        )
        ax.axhline(stationary_mean, color=COLORS["green"], linestyle="--", label="média de longo prazo")
        ax.axvline(observed_time[-1], color=COLORS["gray"], linestyle=":")
        ax.set(title="Forecast probabilístico: 100 futuros possíveis + distribuição analítica", xlabel="tempo", ylabel=r"$X_t$")
        ax.legend(ncol=2)
        plt.show()
        """
    ),
    md(
        r"""
        ### 6.7 A fórmula e a simulação realmente concordam?

        Como conhecemos a distribuição correta neste laboratório, podemos verificar a simulação.
        Com muitas trajetórias, média, desvio-padrão e cobertura Monte Carlo devem se aproximar
        dos valores analíticos. Pequenas diferenças permanecem por erro de amostragem.
        """
    ),
    code(
        """
        mc_mean_error = np.max(np.abs(future_paths.mean(axis=0) - analytic.mean))
        mc_std_error = np.max(np.abs(future_paths.std(axis=0) - analytic.std))
        empirical_coverage = np.mean(
            (future_paths >= analytic.lower) & (future_paths <= analytic.upper), axis=0
        )

        pd.DataFrame({
            "checagem": [
                "erro máx. da média Monte Carlo",
                "erro máx. do desvio Monte Carlo",
                "cobertura média do intervalo 95%",
                "menor cobertura entre horizontes",
                "maior cobertura entre horizontes",
            ],
            "valor": [
                mc_mean_error,
                mc_std_error,
                empirical_coverage.mean(),
                empirical_coverage.min(),
                empirical_coverage.max(),
            ],
        }).round(4)
        """
    ),
    md(
        r"""
        ### 6.8 O que “intervalo preditivo de 95%” significa?

        Para um horizonte específico, por exemplo $h=10$, o intervalo cobre os 95% centrais da
        distribuição de $X_{T+10}\mid X_T$. Se repetíssemos o experimento muitas vezes, cerca de
        95% dos valores realizados em $T+10$ cairiam dentro dos respectivos intervalos.

        Isso **não** significa que:

        - há 95% de chance de a linha da média acontecer — a linha é apenas um resumo;
        - 95% de cada trajetória completa ficará sempre dentro da faixa;
        - os parâmetros $c$, $\phi$ e $\sigma$ são conhecidos com certeza em dados reais.

        A faixa é **marginal por horizonte**, não uma banda simultânea para todo o caminho. Além
        disso, neste laboratório ela representa apenas a incerteza das inovações futuras. Quando
        os parâmetros são estimados, existe uma camada adicional de incerteza que estudaremos em
        notebooks posteriores.
        """
    ),
    md(
        r"""
        ## 7. Exercícios

        Altere os parâmetros e formule sua resposta antes de abrir o gabarito.

        ### Exercício 1 — Classificação

        Para cada $\phi \in \{-1{,}1,-0{,}6,0,0{,}8,1,1{,}05\}$, classifique o processo como
        estacionário, raiz unitária ou explosivo. O que muda quando $\phi$ é negativo?

        ### Exercício 2 — Persistência

        Calcule a meia-vida para $\phi=0{,}7$, $0{,}9$ e $0{,}98$. Em qual caso uma amostra de
        100 pontos pode parecer “quase não estacionária”?

        ### Exercício 3 — Passeio aleatório

        Se $\sigma=2$, qual é a variância teórica de $X_{100}$? E de $X_{400}$? Confirme por
        simulação e compare a razão entre elas.

        ### Exercício 4 — Forecast

        Use $\phi=0{,}6$, $c=0$, $\sigma=1$ e $X_T=3$. Calcule a média e a variância nos
        horizontes 1 e 5. Depois compare a fórmula com 50.000 caminhos simulados.

        ### Exercício 5 — Choque com sinal alternado

        Para $\phi=-0{,}5$ e choque $\delta=4$, escreva a resposta nos horizontes 0 a 4 sem
        executar código. Depois confira abaixo.
        """
    ),
    code(
        """
        # Área de experimentação — mude os valores e execute novamente.
        phi_ex = 0.90
        sigma_ex = 2.0
        horizon_ex = 5

        print(f"meia-vida(phi={phi_ex}) = {half_life(phi_ex):.2f}")
        print("resposta ao choque:", shock_response(-0.5, 4.0, horizon_ex))
        """
    ),
    md(
        r"""
        <details>
        <summary><strong>Gabarito comentado</strong> — abra depois de tentar</summary>

        1. Estacionários: $-0{,}6$, $0$ e $0{,}8$. Raízes unitárias: $-1{,}1$ não é;
           $\phi=1$ é a raiz unitária usual. Explosivos em módulo: $-1{,}1$ e $1{,}05$.
           Com $\phi<0$, os efeitos alternam de sinal.
        2. Meias-vidas aproximadas: 1,94; 6,58; 34,31. O caso $0{,}98$ é o mais persistente.
        3. $\mathrm{Var}(X_{100})=100\times 2^2=400$ e
           $\mathrm{Var}(X_{400})=1600$; a razão é 4.
        4. Média: $3\phi^h$; variância: $\sum_{j=0}^{h-1}\phi^{2j}$.
           Em $h=1$: média 1,8 e variância 1. Em $h=5$: média 0,23328 e variância ≈1,553.
        5. $[4,-2,1,-0{,}5,0{,}25]$.

        </details>
        """
    ),
    code(
        """
        # Gabarito executável: checagens numéricas dos exercícios 2–5.
        np.testing.assert_allclose(
            [half_life(phi) for phi in [0.7, 0.9, 0.98]],
            [1.943358, 6.578813, 34.309618],
            rtol=1e-5,
        )

        rw_ex = simulate_random_walk(400, n_paths=30_000, sigma=2.0, seed=701)
        ratio_ex = rw_ex[:, 400].var() / rw_ex[:, 100].var()
        assert 3.8 < ratio_ex < 4.2

        forecast_ex = forecast_ar1(3.0, 5, phi=0.6, sigma=1.0)
        np.testing.assert_allclose(forecast_ex.mean[[0, 4]], [1.8, 0.23328])
        np.testing.assert_allclose(forecast_ex.std[[0, 4]] ** 2, [1.0, 1.55305216])
        np.testing.assert_allclose(shock_response(-0.5, 4.0, 4), [4, -2, 1, -0.5, 0.25])

        print(f"✓ Gabarito numérico validado (razão simulada de variâncias: {ratio_ex:.3f})")
        """
    ),
    md(
        r"""
        ## 8. Testes rápidos dentro do notebook

        Estes testes não substituem a suíte automatizada do repositório, mas verificam as
        invariantes centrais no mesmo ambiente em que as figuras foram geradas.
        """
    ),
    code(
        """
        # Reprodutibilidade e formas
        np.testing.assert_array_equal(
            simulate_white_noise(20, seed=8), simulate_white_noise(20, seed=8)
        )
        assert simulate_random_walk(10, n_paths=3, seed=8).shape == (3, 11)

        # Fórmulas analíticas
        np.testing.assert_allclose(theoretical_ar1_acf(0.5, 3), [1, 0.5, 0.25, 0.125])
        np.testing.assert_allclose(shock_response(0.5, 2, 3), [2, 1, 0.5, 0.25])
        mean_08, variance_08 = ar1_stationary_moments(phi=0.8, sigma=1)
        np.testing.assert_allclose([mean_08, variance_08], [0, 1 / (1 - 0.8**2)])

        # Monte Carlo versus forecast analítico
        quick_analytic = forecast_ar1(1.0, 8, phi=0.7, sigma=1)
        quick_mc = simulate_ar1_future_paths(1.0, 8, phi=0.7, sigma=1, n_paths=40_000, seed=9)
        np.testing.assert_allclose(quick_mc.mean(axis=0), quick_analytic.mean, atol=0.02)
        np.testing.assert_allclose(quick_mc.std(axis=0), quick_analytic.std, atol=0.025)

        print("✓ Todos os testes rápidos passaram.")
        """
    ),
    md(
        r"""
        ## 9. Síntese e próximos passos

        | Processo | Memória | Variância no tempo | Estacionário? | Forecast |
        |---|---|---|---|---|
        | Ruído branco | nenhuma | constante | sim | média constante, sem ganho do passado |
        | Passeio aleatório | choques permanentes | cresce com $t$ | não | último nível; incerteza cresce sem limite |
        | AR(1), $|\phi|<1$ | decai como $\phi^h$ | converge | sim | reverte à média; incerteza converge |

        ### Armadilhas a evitar

        - confundir uma trajetória visualmente suave com um processo determinístico;
        - usar fórmulas estacionárias quando $|\phi|\ge 1$;
        - tratar intervalo preditivo como intervalo de confiança de parâmetro;
        - interpretar uma ACF alta sem antes verificar tendência, sazonalidade ou quebras.

        ### Extensões sugeridas

        1. adicionar drift ao passeio aleatório;
        2. estimar $\phi$ a partir de dados em vez de defini-lo;
        3. avançar para AR(p), MA(q) e ARMA;
        4. introduzir volatilidade condicional com ARCH/GARCH;
        5. conectar o limite contínuo ao movimento Browniano e ao movimento Browniano geométrico.

        **Ideia-chave:** previsão em séries temporais é uma distribuição condicionada ao que já
        observamos. O ponto previsto é apenas um resumo dessa distribuição.
        """
    ),
]


def main() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(CELLS, start=1):
        cell["id"] = f"nb01-{index:03d}"
    payload["cells"] = CELLS
    payload["metadata"].update(
        {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "time_series_lab": {"notebook": 1, "language": "pt-BR"},
        }
    )
    NOTEBOOK.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Notebook reconstruído: {NOTEBOOK} ({len(CELLS)} células)")


if __name__ == "__main__":
    main()

