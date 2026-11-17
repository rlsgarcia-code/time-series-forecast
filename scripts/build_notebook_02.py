"""Reconstrói o Notebook 02 a partir de células versionáveis neste script."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "02_estimacao_diagnostico_ar1.ipynb"


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
        # Notebook 02 — Estimação e diagnóstico de um AR(1)

        **Do modelo conhecido ao modelo aprendido a partir dos dados**

        No Notebook 01, escolhemos $c$, $\phi$ e $\sigma$ e geramos séries. Agora faremos o
        caminho inverso: observaremos uma série, estimaremos os parâmetros, verificaremos se os
        resíduos parecem ruído branco e avaliaremos forecasts fora da amostra.
        """
    ),
    md(
        r"""
        ## Para quem é este notebook

        **Pré-requisito:** Notebook 01, especialmente AR(1), ACF e forecast probabilístico.

        ### Objetivos de aprendizagem

        Ao final, você será capaz de:

        1. enxergar um AR(1) como uma regressão de $X_t$ sobre $X_{t-1}$;
        2. estimar intercepto, $\phi$ e $\sigma$ por mínimos quadrados;
        3. interpretar erro-padrão e intervalo de confiança dos parâmetros;
        4. entender como tamanho da amostra e persistência afetam a estimação;
        5. diagnosticar resíduos com gráficos, ACF e teste de Ljung–Box;
        6. evitar vazamento temporal ao separar treino e teste;
        7. distinguir forecast multi-step de origem fixa e forecast rolling de um passo.

        ### Roteiro

        1. Gerar uma série e esconder seus parâmetros
        2. Derivar e implementar OLS para AR(1)
        3. Quantificar incerteza de estimação
        4. Diagnosticar resíduos e detectar má especificação
        5. Avaliar forecasts fora da amostra
        6. Exercícios, gabarito e testes rápidos
        """
    ),
    md(
        r"""
        ## Espinha matemática: do processo ao estimador

        O modelo populacional é

        $$X_t=c+\phi X_{t-1}+\varepsilon_t,\qquad
        E[\varepsilon_t\mid\mathcal F_{t-1}]=0,\quad
        \operatorname{Var}(\varepsilon_t\mid\mathcal F_{t-1})=\sigma^2.$$

        Empilhando $t=1,\ldots,T$, obtemos

        $$\mathbf y=\mathbf Z\beta+\boldsymbol\varepsilon,\qquad
        \beta=\begin{bmatrix}c\\\phi\end{bmatrix},\qquad
        \mathbf Z=\begin{bmatrix}1&X_0\\\vdots&\vdots\\1&X_{T-1}\end{bmatrix}.$$

        O estimador de mínimos quadrados e a variância residual são

        $$\widehat\beta=(\mathbf Z^\top\mathbf Z)^{-1}\mathbf Z^\top\mathbf y,$$

        $$\widehat\sigma^2=\frac{1}{T-2}
        (\mathbf y-\mathbf Z\widehat\beta)^\top
        (\mathbf y-\mathbf Z\widehat\beta).$$

        Condicionalmente aos regressores e sob as hipóteses gaussianas,

        $$\widehat\beta\mid\mathbf Z\sim N\!\left(
        \beta,\sigma^2(\mathbf Z^\top\mathbf Z)^{-1}\right).$$

        Em séries temporais, essa expressão deve ser interpretada com cuidado: os regressores são
        aleatórios e a aproximação assintótica se deteriora perto de $\phi=1$. Distinguiremos
        sempre **parâmetro** $(c,\phi,\sigma)$, **estimativa**
        $(\widehat c,\widehat\phi,\widehat\sigma)$ e **variável futura** $X_{T+h}$.
        """
    ),
    md(
        r"""
        ## 0. Preparação

        Todas as funções matemáticas usadas no fluxo são definidas explicitamente no notebook.
        Sementes fixas tornam as comparações reproduzíveis.
        """
    ),
    code(
        """
        from dataclasses import dataclass
        from importlib.metadata import version

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy import stats
        from statsmodels.stats.diagnostic import acorr_ljungbox

        %matplotlib inline

        COLORS = {
            "blue": "#2563EB", "orange": "#EA580C", "green": "#059669",
            "purple": "#7C3AED", "gray": "#64748B", "red": "#DC2626",
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
            {"versão": [version(pkg) for pkg in ["numpy", "pandas", "matplotlib", "statsmodels"]]},
            index=["numpy", "pandas", "matplotlib", "statsmodels"],
        )
        """
    ),
    md(
        r"""
        ## 1. O experimento: observar os dados, não os parâmetros

        O processo verdadeiro será

        $$X_t=c+\phi X_{t-1}+\varepsilon_t,\qquad
        \varepsilon_t\sim\mathcal N(0,\sigma^2).$$

        Usaremos os parâmetros verdadeiros apenas para gerar os dados. Durante a estimação,
        agiremos como se eles fossem desconhecidos — exatamente como em um problema real.
        """
    ),
    code(
        """
        def simulate_ar1(
            n_steps, *, phi, intercept=0.0, sigma=1.0,
            x0=0.0, n_paths=1, seed=None
        ):
            '''Simula X_t = c + phi·X_(t-1) + ε_t. Inclui X_0.'''
            rng = np.random.default_rng(seed)
            epsilon = rng.normal(0.0, sigma, size=(n_paths, n_steps))
            paths = np.empty((n_paths, n_steps + 1))
            paths[:, 0] = x0
            for t in range(1, n_steps + 1):
                paths[:, t] = (
                    intercept + phi * paths[:, t - 1] + epsilon[:, t - 1]
                )
            return paths
        """
    ),
    code(
        """
        TRUE = {"intercept": 0.40, "phi": 0.72, "sigma": 0.85}
        long_run_mean = TRUE["intercept"] / (1 - TRUE["phi"])
        series = simulate_ar1(
            500,
            **TRUE,
            x0=long_run_mean,
            seed=1001,
        )[0]

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(series, color=COLORS["blue"], linewidth=1)
        ax.axhline(long_run_mean, color=COLORS["green"], linestyle="--", label="média verdadeira")
        ax.set(title="A única série que o estimador enxerga", xlabel="tempo", ylabel=r"$X_t$")
        ax.legend()
        plt.show()
        """
    ),
    md(
        r"""
        ## 2. AR(1) como regressão

        Para cada $t=1,\ldots,T$, formamos um par:

        $$\underbrace{X_t}_{y_t}
        =c+\phi\underbrace{X_{t-1}}_{x_t}+\varepsilon_t.$$

        Assim, estimar um AR(1) equivale a ajustar uma reta aos pontos
        $(X_{t-1},X_t)$. O intercepto da reta estima $c$ e sua inclinação estima $\phi$.

        **Cuidado:** essa nuvem contém pares temporais consecutivos; ela não transforma o
        problema em dados independentes comuns. A interpretação continua sendo temporal.
        """
    ),
    code(
        """
        lagged = series[:-1]
        response = series[1:]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(lagged, response, s=18, alpha=0.35, color=COLORS["blue"])
        ax.set(
            title=r"Pares consecutivos: $X_{t-1}$ versus $X_t$",
            xlabel=r"regressor $X_{t-1}$",
            ylabel=r"resposta $X_t$",
        )
        plt.show()
        """
    ),
    md(
        r"""
        ### 2.1 Mínimos quadrados ordinários

        Empilhamos os dados na forma matricial $\mathbf y=\mathbf X\boldsymbol\beta+\boldsymbol\varepsilon$,
        com $\boldsymbol\beta=(c,\phi)'$. O estimador OLS minimiza a soma dos quadrados:

        $$\widehat{\boldsymbol\beta}
        =\arg\min_{\boldsymbol\beta}\|\mathbf y-\mathbf X\boldsymbol\beta\|^2
        =(\mathbf X'\mathbf X)^{-1}\mathbf X'\mathbf y.$$

        Numericamente, `np.linalg.lstsq` resolve o mesmo problema de forma mais estável do que
        calcular a inversa diretamente.
        """
    ),
    code(
        """
        @dataclass(frozen=True)
        class AR1Fit:
            intercept: float
            phi: float
            sigma: float
            standard_errors: np.ndarray
            covariance: np.ndarray
            fitted_values: np.ndarray
            residuals: np.ndarray
            nobs: int
            degrees_of_freedom: int

            @property
            def parameters(self):
                return np.array([self.intercept, self.phi])


        def fit_ar1_ols(values):
            '''Estima c e phi em X_t = c + phi·X_(t-1) + erro_t.'''
            x = np.asarray(values, dtype=float)
            y = x[1:]
            x_lag = x[:-1]
            design = np.column_stack([np.ones_like(x_lag), x_lag])

            beta_hat = np.linalg.lstsq(design, y, rcond=None)[0]
            fitted = design @ beta_hat
            residuals = y - fitted

            # Dois parâmetros foram estimados: graus de liberdade = (T - 1) - 2.
            dof = y.size - design.shape[1]
            residual_variance = residuals @ residuals / dof
            covariance = residual_variance * np.linalg.inv(design.T @ design)

            return AR1Fit(
                intercept=float(beta_hat[0]),
                phi=float(beta_hat[1]),
                sigma=float(np.sqrt(residual_variance)),
                standard_errors=np.sqrt(np.diag(covariance)),
                covariance=covariance,
                fitted_values=fitted,
                residuals=residuals,
                nobs=y.size,
                degrees_of_freedom=dof,
            )
        """
    ),
    code(
        """
        fit = fit_ar1_ols(series)

        parameter_table = pd.DataFrame({
            "parâmetro": ["intercepto c", "persistência phi", "desvio sigma"],
            "verdadeiro": [TRUE["intercept"], TRUE["phi"], TRUE["sigma"]],
            "estimado": [fit.intercept, fit.phi, fit.sigma],
        })
        parameter_table["erro de estimação"] = parameter_table["estimado"] - parameter_table["verdadeiro"]
        parameter_table.round(4)
        """
    ),
    code(
        """
        # A reta estimada sobre a nuvem de pares consecutivos.
        grid = np.linspace(lagged.min(), lagged.max(), 200)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(lagged, response, s=18, alpha=0.25, color=COLORS["blue"], label="dados")
        ax.plot(grid, fit.intercept + fit.phi * grid, color=COLORS["orange"], linewidth=2.5, label="reta estimada")
        ax.plot(grid, TRUE["intercept"] + TRUE["phi"] * grid, color=COLORS["green"], linestyle="--", label="reta verdadeira")
        ax.set(title="Estimativa versus mecanismo verdadeiro", xlabel=r"$X_{t-1}$", ylabel=r"$X_t$")
        ax.legend()
        plt.show()
        """
    ),
    md(
        r"""
        A reta estimada não coincide exatamente com a verdadeira porque dispomos de uma amostra
        finita. Isso não é uma falha: estimadores são variáveis aleatórias e mudariam se outra
        trajetória fosse observada.

        Os resíduos são

        $$\widehat\varepsilon_t=X_t-\widehat c-\widehat\phi X_{t-1}.$$

        O estimador de $\sigma$ é o desvio-padrão desses resíduos, com correção pelos dois
        parâmetros estimados.
        """
    ),
    md(
        r"""
        ## 3. Incerteza dos parâmetros

        Sob as hipóteses do AR(1) gaussiano, a covariância OLS aproximada é

        $$\widehat{\mathrm{Var}}(\widehat{\boldsymbol\beta})
        =\widehat\sigma^2(\mathbf X'\mathbf X)^{-1}.$$

        A raiz da diagonal fornece os erros-padrão de $\widehat c$ e $\widehat\phi$. Um intervalo
        aproximado de 95% é estimativa $\pm 1{,}96\times$ erro-padrão.
        """
    ),
    code(
        """
        def parameter_intervals(fit, confidence=0.95):
            '''Intervalos normais aproximados para c e phi.'''
            z = stats.norm.ppf((1 + confidence) / 2)
            margin = z * fit.standard_errors
            return fit.parameters - margin, fit.parameters + margin


        lower, upper = parameter_intervals(fit)
        pd.DataFrame({
            "parâmetro": ["c", "phi"],
            "estimativa": fit.parameters,
            "erro-padrão": fit.standard_errors,
            "limite 2,5%": lower,
            "limite 97,5%": upper,
            "valor verdadeiro": [TRUE["intercept"], TRUE["phi"]],
        }).round(4)
        """
    ),
    md(
        r"""
        **Interpretação frequencista:** o parâmetro verdadeiro é fixo. Se repetíssemos o processo
        de coletar uma série e construir o intervalo, aproximadamente 95% desses intervalos
        conteriam o valor verdadeiro. Não dizemos que este intervalo específico tem 95% de
        probabilidade de conter um parâmetro aleatório.

        ### 3.1 O efeito do tamanho da amostra

        Repetiremos o experimento muitas vezes. Cada série produz um $\widehat\phi$ diferente.
        À medida que $n$ cresce, a distribuição amostral deve se concentrar em torno de $\phi$.
        """
    ),
    code(
        """
        def repeated_phi_estimates(n, *, phi, intercept, sigma, repetitions=700, seed=0):
            '''Gera muitas séries e retorna uma estimativa de phi para cada uma.'''
            rng = np.random.default_rng(seed)
            estimates = np.empty(repetitions)
            mean = intercept / (1 - phi)
            for i in range(repetitions):
                path = simulate_ar1(
                    n - 1, phi=phi, intercept=intercept, sigma=sigma,
                    x0=mean, seed=int(rng.integers(0, 2**32 - 1))
                )[0]
                estimates[i] = fit_ar1_ols(path).phi
            return estimates


        sample_sizes = [30, 100, 500]
        phi_distributions = {
            n: repeated_phi_estimates(n, **TRUE, seed=1100 + n)
            for n in sample_sizes
        }

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
        for ax, n in zip(axes, sample_sizes):
            values = phi_distributions[n]
            ax.hist(values, bins=28, density=True, color=COLORS["blue"], alpha=0.75)
            ax.axvline(TRUE["phi"], color=COLORS["green"], linestyle="--", linewidth=2)
            ax.axvline(values.mean(), color=COLORS["orange"], linewidth=2)
            ax.set_title(f"n={n}\\nDP={values.std():.3f}")
            ax.set_xlabel(r"$\\widehat\\phi$")
        axes[0].set_ylabel("densidade")
        fig.suptitle("Mais dados reduzem a dispersão do estimador", fontweight="bold")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        r"""
        ### 3.2 Persistência alta e viés em amostras pequenas

        Quando $\phi$ está próximo de 1, séries curtas frequentemente produzem estimativas
        enviesadas para baixo. Esse é um problema de amostra finita: a aproximação assintótica
        melhora com mais dados, mas pode ser otimista justamente nos casos mais persistentes.
        """
    ),
    code(
        """
        comparison = []
        for phi_value in [0.60, 0.95]:
            estimates = repeated_phi_estimates(
                50, phi=phi_value, intercept=0.0, sigma=1.0,
                repetitions=1_000, seed=1200 + int(phi_value * 100)
            )
            comparison.append(pd.DataFrame({"phi verdadeiro": phi_value, "phi estimado": estimates}))
        bias_data = pd.concat(comparison, ignore_index=True)

        summary_bias = bias_data.groupby("phi verdadeiro")["phi estimado"].agg(["mean", "std"])
        summary_bias["viés"] = summary_bias["mean"] - summary_bias.index
        summary_bias.round(4)
        """
    ),
    md(
        r"""
        ## 4. Diagnóstico: os resíduos parecem ruído branco?

        Estimar parâmetros não encerra a modelagem. Se o AR(1) capturou a dependência temporal,
        os resíduos devem se comportar como inovações:

        - média próxima de zero;
        - variância aproximadamente constante;
        - ausência de autocorrelação remanescente;
        - distribuição aproximadamente normal, se quisermos intervalos gaussianos.

        Um bom ajuste visual da série não garante resíduos adequados.
        """
    ),
    code(
        """
        def sample_acf(values, nlags):
            x = np.asarray(values, dtype=float)
            centered = x - x.mean()
            denominator = centered @ centered
            acf = np.empty(nlags + 1)
            acf[0] = 1.0
            for lag in range(1, nlags + 1):
                acf[lag] = centered[lag:] @ centered[:-lag] / denominator
            return acf


        residual_acf = sample_acf(fit.residuals, 24)
        normal_quantiles, ordered_residuals = stats.probplot(fit.residuals, dist="norm", fit=False)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(fit.residuals, color=COLORS["blue"], linewidth=1)
        axes[0, 0].axhline(0, color="black", linewidth=0.8)
        axes[0, 0].set(title="Resíduos no tempo", xlabel="tempo", ylabel="resíduo")

        axes[0, 1].hist(fit.residuals, bins=28, density=True, color=COLORS["blue"], alpha=0.7)
        grid_resid = np.linspace(fit.residuals.min(), fit.residuals.max(), 200)
        axes[0, 1].plot(grid_resid, stats.norm.pdf(grid_resid, 0, fit.sigma), color=COLORS["orange"])
        axes[0, 1].set(title="Histograma versus normal ajustada", xlabel="resíduo")

        axes[1, 0].vlines(np.arange(25), 0, residual_acf, color=COLORS["purple"], linewidth=3)
        bound = 1.96 / np.sqrt(fit.nobs)
        axes[1, 0].axhspan(-bound, bound, color=COLORS["gray"], alpha=0.15)
        axes[1, 0].axhline(0, color="black", linewidth=0.8)
        axes[1, 0].set(title="ACF dos resíduos", xlabel="lag", ylabel="autocorrelação")

        axes[1, 1].scatter(normal_quantiles, ordered_residuals, s=18, alpha=0.5, color=COLORS["blue"])
        slope, intercept_qq = np.polyfit(normal_quantiles, ordered_residuals, 1)
        axes[1, 1].plot(normal_quantiles, intercept_qq + slope * np.asarray(normal_quantiles), color=COLORS["orange"])
        axes[1, 1].set(title="Q–Q plot normal", xlabel="quantis normais", ylabel="resíduos ordenados")

        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        r"""
        ### 4.1 Teste de Ljung–Box

        O teste combina as autocorrelações residuais até o lag $m$:

        $$Q(m)=n(n+2)\sum_{k=1}^{m}\frac{\widehat\rho_k^2}{n-k}.$$

        - $H_0$: não há autocorrelação residual até o lag $m$;
        - p-valor pequeno: evidência de dependência ainda não explicada;
        - p-valor grande: não rejeitar $H_0$ **não prova** independência, apenas não detecta
          evidência suficiente contra ela.

        Testamos vários lags para evitar depender de uma única escolha arbitrária.
        """
    ),
    code(
        """
        ljung_box_good = acorr_ljungbox(
            fit.residuals, lags=[5, 10, 20], return_df=True, model_df=2
        )
        ljung_box_good.index.name = "lag"
        ljung_box_good.rename(columns={"lb_stat": "estatística Q", "lb_pvalue": "p-valor"}).round(4)
        """
    ),
    md(
        r"""
        ### 4.2 Um contraste: ajustar AR(1) a dados AR(2)

        Para entender o diagnóstico, criaremos deliberadamente um modelo errado. Os dados seguem

        $$X_t=0{,}75X_{t-1}-0{,}35X_{t-2}+\varepsilon_t,$$

        mas ajustaremos apenas um AR(1). Se o diagnóstico funciona, a dependência omitida deve
        aparecer na ACF dos resíduos e no Ljung–Box.
        """
    ),
    code(
        """
        def simulate_ar2(n_steps, *, phi1, phi2, sigma=1.0, seed=None):
            rng = np.random.default_rng(seed)
            epsilon = rng.normal(0, sigma, n_steps)
            x = np.zeros(n_steps + 2)
            for t in range(2, n_steps + 2):
                x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + epsilon[t - 2]
            return x[2:]


        ar2_series = simulate_ar2(700, phi1=0.75, phi2=-0.35, sigma=0.85, seed=1301)
        wrong_fit = fit_ar1_ols(ar2_series)
        wrong_acf = sample_acf(wrong_fit.residuals, 24)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, acf_values, title, color in [
            (axes[0], residual_acf, "AR(1) correto", COLORS["green"]),
            (axes[1], wrong_acf, "AR(1) ajustado a dados AR(2)", COLORS["red"]),
        ]:
            ax.vlines(np.arange(25), 0, acf_values, color=color, linewidth=3)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set(title=title, xlabel="lag", ylabel="ACF residual", ylim=(-0.45, 1.05))
        plt.tight_layout()
        plt.show()

        acorr_ljungbox(wrong_fit.residuals, lags=[5, 10, 20], return_df=True, model_df=2).round(5)
        """
    ),
    md(
        r"""
        A rejeição não diz automaticamente qual modelo usar; diz que o AR(1) deixou estrutura
        previsível nos resíduos. A ACF e o conhecimento do domínio ajudam a propor a próxima
        especificação — neste caso, incluir uma segunda defasagem.
        """
    ),
    md(
        r"""
        ## 5. Avaliação fora da amostra

        Métricas dentro da amostra avaliam ajuste; forecast exige dados que o modelo não viu.
        Dividiremos a série **preservando a ordem temporal**:

        - treino: primeiras observações, usadas para estimar;
        - teste: observações finais, usadas somente para avaliar.

        Em séries temporais, embaralhar aleatoriamente pode deixar o futuro vazar para o treino.
        """
    ),
    code(
        """
        evaluation_series = simulate_ar1(
            360, **TRUE, x0=long_run_mean, seed=1401
        )[0]
        train_size = 280
        train = evaluation_series[:train_size]
        test = evaluation_series[train_size:]

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(np.arange(train.size), train, color=COLORS["blue"], label="treino")
        ax.plot(np.arange(train.size, evaluation_series.size), test, color=COLORS["orange"], label="teste")
        ax.axvline(train.size - 1, color=COLORS["gray"], linestyle="--")
        ax.set(title="Divisão temporal: nenhuma observação futura entra no treino", xlabel="tempo", ylabel=r"$X_t$")
        ax.legend()
        plt.show()
        """
    ),
    code(
        """
        train_fit = fit_ar1_ols(train)
        pd.DataFrame({
            "parâmetro": ["c", "phi", "sigma"],
            "estimado no treino": [train_fit.intercept, train_fit.phi, train_fit.sigma],
            "verdadeiro": [TRUE["intercept"], TRUE["phi"], TRUE["sigma"]],
        }).round(4)
        """
    ),
    md(
        r"""
        ### 5.1 Forecast multi-step de origem fixa

        Estimamos uma vez no fim do treino e prevemos todo o bloco de teste sem observar seus
        valores intermediários. Para $h>1$, a previsão usa recursivamente suas próprias médias.

        Os intervalos abaixo são **plug-in**: substituem parâmetros desconhecidos por suas
        estimativas e contabilizam as inovações futuras, mas ignoram a incerteza em
        $\widehat c$, $\widehat\phi$ e $\widehat\sigma$.
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


        def forecast_ar1(last_value, horizon, *, intercept, phi, sigma, confidence=0.95):
            h = np.arange(1, horizon + 1)
            mean = phi**h * last_value + intercept * (1 - phi**h) / (1 - phi)
            variance = sigma**2 * np.cumsum(phi ** (2 * np.arange(horizon)))
            std = np.sqrt(variance)
            z = stats.norm.ppf((1 + confidence) / 2)
            return ForecastResult(h, mean, std, mean - z * std, mean + z * std)


        fixed_forecast = forecast_ar1(
            train[-1], test.size,
            intercept=train_fit.intercept,
            phi=train_fit.phi,
            sigma=train_fit.sigma,
        )
        """
    ),
    md(
        r"""
        #### 5.1.1 Leitura inferencial da função, linha por linha

        Depois da estimação, chamamos a função com
        $(\widehat c,\widehat\phi,\widehat\sigma)$ e obtemos, para cada $h$,

        $$X_{T+h}\mid X_T,\widehat c,\widehat\phi,\widehat\sigma
        \approx\mathcal N(\widehat m_h,\widehat s_h^2).$$

        | Código | Quantidade matemática | Interpretação |
        |---|---|---|
        | `h = np.arange(1, horizon + 1)` | $h=1,\ldots,H$ | generaliza um passo para todos os horizontes |
        | `phi**h * last_value` | $\widehat\phi^hX_T$ | memória restante da última observação |
        | `intercept*(1-phi**h)/(1-phi)` | $\widehat c\sum_{j=0}^{h-1}\widehat\phi^j$ | interceptos acumulados |
        | `sigma**2*cumsum(phi**(2*j))` | $\widehat\sigma^2\sum_{j=0}^{h-1}\widehat\phi^{2j}$ | variância acumulada dos choques futuros |
        | `sqrt(variance)` | $\widehat s_h$ | erro-padrão preditivo na unidade de $X$ |
        | `norm.ppf((1+confidence)/2)` | $z_{1-\alpha/2}$ | quantil bilateral para cobertura $1-\alpha$ |
        | `mean ± z*std` | intervalo preditivo | faixa para uma nova observação $X_{T+h}$ |

        A soma usa $\phi^{2j}$ porque variâncias recebem o **quadrado** dos pesos. Não seria correto
        somar desvios-padrão: as inovações futuras são independentes e suas variâncias se somam.

        #### Intervalo preditivo não é intervalo de confiança do parâmetro

        - O intervalo de confiança de $\phi$ no capítulo 3 responde “quais valores de $\phi$ são
          compatíveis com a amostra?”.
        - O intervalo acima responde “quais valores uma nova observação futura pode assumir?”.

        Pela lei da variância total, uma inferência que incorpora parâmetros estimados teria

        $$\mathrm{Var}(X_{T+h}\mid\text{dados})=
        \mathbb E_{\theta\mid dados}[\mathrm{Var}(X_{T+h}\mid\theta,dados)]
        +\mathrm{Var}_{\theta\mid dados}[\mathbb E(X_{T+h}\mid\theta,dados)].$$

        A função plug-in calcula essencialmente o primeiro termo em
        $\theta=\widehat\theta$. Bootstrap paramétrico ou inferência Bayesiana também propagariam o
        segundo. A aproximação tende a melhorar com amostras grandes, mas merece cautela quando
        $\widehat\phi$ está perto de 1. A fórmula escrita assume $\phi\neq1$; na raiz unitária,
        usamos média $X_T+hc$ e variância $h\sigma^2$.
        """
    ),
    code(
        """
        test_time = np.arange(train.size, evaluation_series.size)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(np.arange(train.size - 60, train.size), train[-60:], color="black", label="treino observado")
        ax.plot(test_time, test, color=COLORS["blue"], linewidth=2, label="teste realizado")
        ax.plot(test_time, fixed_forecast.mean, color=COLORS["orange"], linewidth=2, label="média prevista")
        ax.fill_between(test_time, fixed_forecast.lower, fixed_forecast.upper, color=COLORS["orange"], alpha=0.18, label="intervalo plug-in 95%")
        ax.axvline(train.size - 1, color=COLORS["gray"], linestyle="--")
        ax.set(title="Forecast multi-step de origem fixa", xlabel="tempo", ylabel=r"$X_t$")
        ax.legend(ncol=2)
        plt.show()
        """
    ),
    code(
        """
        def forecast_metrics(actual, predicted, *, lower=None, upper=None):
            actual = np.asarray(actual)
            predicted = np.asarray(predicted)
            errors = actual - predicted
            result = {
                "MAE": np.mean(np.abs(errors)),
                "RMSE": np.sqrt(np.mean(errors**2)),
            }
            if lower is not None and upper is not None:
                result["cobertura"] = np.mean((actual >= lower) & (actual <= upper))
                result["largura média"] = np.mean(upper - lower)
            return result


        pd.Series(forecast_metrics(
            test,
            fixed_forecast.mean,
            lower=fixed_forecast.lower,
            upper=fixed_forecast.upper,
        ), name="origem fixa").to_frame().round(4)
        """
    ),
    md(
        r"""
        Uma cobertura observada diferente de 95% em apenas um bloco de teste não invalida o
        modelo: cobertura é uma propriedade de repetição. Também não devemos escolher modelo
        apenas por cobertura — intervalos arbitrariamente largos cobrem muito, mas são pouco úteis.

        ### 5.2 Forecast rolling de um passo

        Em produção, muitas vezes recebemos o valor real seguinte, incorporamos a nova informação
        e prevemos mais um passo. No esquema rolling:

        1. ajustamos usando tudo até $t$;
        2. prevemos $t+1$;
        3. observamos o valor real de $t+1$;
        4. expandimos a janela e repetimos.

        Compararemos com o baseline ingênuo $\widehat X_{t+1}=X_t$. Um modelo só é útil se superar
        alternativas simples no problema e horizonte relevantes.
        """
    ),
    code(
        """
        def rolling_one_step_ar1(values, initial_train_size):
            '''Reestima com uma janela expansiva e prevê exatamente um passo.'''
            x = np.asarray(values)
            predictions = []
            for target in range(initial_train_size, x.size):
                rolling_fit = fit_ar1_ols(x[:target])
                prediction = rolling_fit.intercept + rolling_fit.phi * x[target - 1]
                predictions.append(prediction)
            return np.asarray(predictions)


        rolling_predictions = rolling_one_step_ar1(evaluation_series, train_size)
        naive_predictions = evaluation_series[train_size - 1:-1]

        rolling_metrics = forecast_metrics(test, rolling_predictions)
        naive_metrics = forecast_metrics(test, naive_predictions)
        pd.DataFrame({"AR(1) rolling": rolling_metrics, "Ingênuo": naive_metrics}).round(4)
        """
    ),
    code(
        """
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(test_time[:35], test[:35], color="black", linewidth=2, label="realizado")
        ax.plot(test_time[:35], rolling_predictions[:35], color=COLORS["orange"], label="AR(1) rolling")
        ax.plot(test_time[:35], naive_predictions[:35], color=COLORS["gray"], linestyle="--", label="ingênuo")
        ax.set(title="Primeiros 35 forecasts rolling de um passo", xlabel="tempo", ylabel=r"$X_t$")
        ax.legend()
        plt.show()
        """
    ),
    md(
        r"""
        **Não compare diretamente as métricas rolling e origem fixa como se fossem o mesmo
        problema.** No rolling, cada previsão recebe observações recentes; na origem fixa, todo o
        caminho futuro é desconhecido. O horizonte e o protocolo de informação fazem parte da
        definição do forecast.

        ## 6. Exercícios

        ### Exercício 1 — Monte os pares

        Para a série `[2.0, 2.5, 1.8, 2.2]`, escreva `y`, `x_lag` e a matriz de desenho com
        intercepto. Quantas linhas entram na regressão?

        ### Exercício 2 — Tamanho amostral

        Compare a dispersão de $\widehat\phi$ para `n=40` e `n=400`, mantendo o mesmo processo.
        Explique por que as estimativas não ficam idênticas ao valor verdadeiro.

        ### Exercício 3 — Diagnóstico

        Se a ACF dos resíduos mostra um pico forte no lag 2 e o Ljung–Box tem p-valor 0,002,
        qual hipótese foi questionada? Isso prova que um AR(2) é o modelo correto?

        ### Exercício 4 — Vazamento

        Explique por que embaralhar uma série antes da divisão treino/teste pode produzir uma
        avaliação otimista.

        ### Exercício 5 — Protocolos

        Dê um exemplo prático em que forecast rolling de um passo é adequado e outro em que uma
        trajetória multi-step de origem fixa é necessária.
        """
    ),
    code(
        """
        # Área de experimentação.
        exercise_series = np.array([2.0, 2.5, 1.8, 2.2])
        exercise_y = exercise_series[1:]
        exercise_x_lag = exercise_series[:-1]
        exercise_design = np.column_stack([np.ones(3), exercise_x_lag])

        print("y =", exercise_y)
        print("x_lag =", exercise_x_lag)
        print("matriz de desenho =\\n", exercise_design)
        """
    ),
    md(
        r"""
        <details>
        <summary><strong>Gabarito comentado</strong> — abra depois de tentar</summary>

        1. Há três pares: `y=[2.5,1.8,2.2]`, `x_lag=[2.0,2.5,1.8]` e a primeira coluna da
           matriz de desenho contém uns para estimar o intercepto.
        2. Com `n=400`, a distribuição é mais estreita, mas cada amostra ainda contém inovações
           aleatórias; consistência não significa igualdade exata em amostra finita.
        3. Foi questionada a hipótese de ausência de autocorrelação residual. O resultado indica
           má especificação, mas não identifica sozinho o modelo correto.
        4. O embaralhamento permite que padrões do futuro influenciem o ajuste e destrói o
           protocolo real de informação disponível no instante da previsão.
        5. Rolling: previsão de demanda atualizada diariamente. Origem fixa: orçamento ou cenário
           de capacidade para os próximos 12 meses antes que qualquer mês futuro seja observado.

        </details>
        """
    ),
    code(
        """
        # Gabarito executável.
        np.testing.assert_array_equal(exercise_y, [2.5, 1.8, 2.2])
        np.testing.assert_array_equal(exercise_x_lag, [2.0, 2.5, 1.8])
        np.testing.assert_array_equal(
            exercise_design,
            [[1.0, 2.0], [1.0, 2.5], [1.0, 1.8]],
        )
        assert phi_distributions[500].std() < phi_distributions[30].std()
        print("✓ Gabarito numérico validado.")
        """
    ),
    md(
        r"""
        ## 7. Testes rápidos dentro do notebook

        Verificamos propriedades algébricas, não números escolhidos por conveniência.
        """
    ),
    code(
        """
        # OLS torna os resíduos ortogonais às colunas da matriz de desenho.
        np.testing.assert_allclose(fit.residuals.sum(), 0.0, atol=1e-10)
        np.testing.assert_allclose(fit.residuals @ series[:-1], 0.0, atol=1e-10)

        # A estimativa recupera o mecanismo aproximadamente em amostra longa.
        long_series = simulate_ar1(
            20_000, **TRUE, x0=long_run_mean, seed=1701
        )[0]
        long_fit = fit_ar1_ols(long_series)
        np.testing.assert_allclose(long_fit.intercept, TRUE["intercept"], atol=0.03)
        np.testing.assert_allclose(long_fit.phi, TRUE["phi"], atol=0.015)
        np.testing.assert_allclose(long_fit.sigma, TRUE["sigma"], atol=0.015)

        # Rolling produz uma previsão para cada observação de teste, sem usar o alvo atual.
        assert rolling_predictions.shape == test.shape
        print("✓ Todos os testes rápidos passaram.")
        """
    ),
    md(
        r"""
        ## 8. Síntese e próximos passos

        | Etapa | Pergunta | Evidência principal |
        |---|---|---|
        | Estimação | Quais parâmetros explicam os pares consecutivos? | $\widehat c$, $\widehat\phi$, $\widehat\sigma$ |
        | Incerteza | Quanto as estimativas mudariam em outra amostra? | erros-padrão, intervalos, Monte Carlo |
        | Diagnóstico | Sobrou estrutura previsível? | resíduos, ACF, Ljung–Box |
        | Avaliação | O modelo prevê dados não vistos? | split temporal, MAE, RMSE, cobertura |
        | Comparação | O modelo supera uma regra simples? | baseline ingênuo sob o mesmo protocolo |

        ### Armadilhas a evitar

        - tratar a estimativa como se fosse o parâmetro verdadeiro;
        - usar intervalos assintóticos sem cautela em amostras pequenas ou $\phi\approx1$;
        - escolher o modelo apenas pelo ajuste dentro da amostra;
        - embaralhar observações antes da validação;
        - interpretar p-valor grande como prova de modelo correto;
        - comparar forecasts de protocolos ou horizontes diferentes.

        **Próximo passo sugerido:** ampliar a memória com AR(p), introduzir choques defasados com
        MA(q) e organizar ambos em modelos ARMA/ARIMA.
        """
    ),
]


def main() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(CELLS, start=1):
        cell["id"] = f"nb02-{index:03d}"
    payload["cells"] = CELLS
    payload["metadata"].update(
        {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "time_series_lab": {"notebook": 2, "language": "pt-BR"},
        }
    )
    NOTEBOOK.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Notebook reconstruído: {NOTEBOOK} ({len(CELLS)} células)")


if __name__ == "__main__":
    main()

