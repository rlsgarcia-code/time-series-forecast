"""Geração explicável e limitada de candidatos a partir da caracterização."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import (
    AR1Spec,
    ARIMASpec,
    GARCH11Spec,
    LocalLevelSpec,
    ModelSpec,
    SeriesProfile,
    TwoRegimeGaussianSpec,
    ValidationError,
    content_id,
)


CandidateMode = Literal["auto", "always", "never"]
DifferencingMode = Literal["auto", "compare", "never"]


@dataclass(frozen=True)
class CandidateDecision:
    """Uma decisão auditável tomada durante a construção do conjunto candidato."""

    status: Literal["included", "excluded"]
    rule: str
    family: str
    spec: ModelSpec | None
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def specification(self) -> str:
        return repr(self.spec) if self.spec is not None else "—"


@dataclass(frozen=True)
class CandidatePlan:
    """Candidatos propostos junto com a trilha completa de decisão."""

    plan_id: str
    series_id: str
    candidates: tuple[ModelSpec, ...]
    decisions: tuple[CandidateDecision, ...]
    summary: Mapping[str, Any]
    policy: Mapping[str, Any]

    def to_frame(self, *, include_excluded: bool = True) -> pd.DataFrame:
        decisions = (
            self.decisions
            if include_excluded
            else tuple(item for item in self.decisions if item.status == "included")
        )
        return pd.DataFrame(
            [
                {
                    "status": item.status,
                    "rule": item.rule,
                    "family": item.family,
                    "specification": item.specification,
                    "reason": item.reason,
                    "evidence": "; ".join(
                        f"{key}={value}" for key, value in item.evidence.items()
                    ),
                }
                for item in decisions
            ]
        )

    def explain(self) -> str:
        lines = [
            (
                f"Plano {self.plan_id} — série {self.series_id}: "
                f"{len(self.candidates)} candidatos."
            ),
            (
                "Estacionariedade: "
                f"{self.summary['stationarity_assessment']}; "
                f"limite ACF/PACF={self.summary['correlation_threshold']:.3f}."
            ),
        ]
        for item in self.decisions:
            symbol = "+" if item.status == "included" else "−"
            lines.append(f"{symbol} {item.specification}: {item.reason}")
        return "\n".join(lines)


@dataclass(frozen=True)
class CandidatePolicy:
    """Converte evidências em hipóteses para backtest, nunca em vencedor automático.

    A política mantém uma grade pequena porque a escolha entre muitos modelos no mesmo
    backtest também pode sobreajustar. A caracterização inclui/exclui hipóteses; o CRPS e os
    guardrails continuam responsáveis pela seleção final.
    """

    significance_level: float = 0.05
    max_ar_order: int = 2
    max_ma_order: int = 2
    max_total_order: int = 3
    max_candidates: int = 14
    differencing: DifferencingMode = "auto"
    garch: CandidateMode = "auto"
    markov_switching: CandidateMode = "auto"
    minimum_garch_observations: int = 40
    # Dois regimes gaussianos têm 6 parâmetros livres: 2 médias, 2 escalas e
    # 2 probabilidades de transição. Seis observações por parâmetro dão 36.
    minimum_markov_observations: int = 36
    change_ratio_threshold: float = 0.15
    simulation_paths: int = 300

    def __post_init__(self) -> None:
        if not 0 < self.significance_level < 0.5:
            raise ValidationError("significance_level deve estar entre 0 e 0.5")
        if self.max_ar_order < 1 or self.max_ma_order < 1:
            raise ValidationError("ordens AR/MA máximas devem ser positivas")
        if self.max_total_order < 1 or self.max_candidates < 4:
            raise ValidationError("limites da política de candidatos são inválidos")
        if self.differencing not in {"auto", "compare", "never"}:
            raise ValidationError("differencing deve ser auto, compare ou never")
        if self.garch not in {"auto", "always", "never"}:
            raise ValidationError("garch deve ser auto, always ou never")
        if self.markov_switching not in {"auto", "always", "never"}:
            raise ValidationError("markov_switching deve ser auto, always ou never")
        if self.minimum_garch_observations < 30:
            raise ValidationError("GARCH exige mínimo de ao menos 30 observações")
        if self.minimum_markov_observations < 20:
            raise ValidationError("mínimo de Markov switching é muito pequeno")
        if self.change_ratio_threshold < 0 or self.simulation_paths < 100:
            raise ValidationError("limiar de mudança/simulações inválido")

    def propose(
        self,
        profile: SeriesProfile,
        *,
        series_kind: str = "level",
        available_families: Sequence[str] = (
            "ar1",
            "arima",
            "local_level",
            "garch11",
            "markov_switching",
        ),
    ) -> CandidatePlan:
        if series_kind not in {"level", "returns"}:
            raise ValidationError("series_kind deve ser level ou returns")

        available = set(available_families)
        decisions: list[CandidateDecision] = []
        candidates: list[ModelSpec] = []
        seen: set[ModelSpec] = set()

        threshold = NormalDist().inv_cdf(1 - self.significance_level / 2) / np.sqrt(
            profile.nobs
        )
        stationarity = self._stationarity_assessment(profile)
        level_p = self._suggested_order(profile.pacf, self.max_ar_order, threshold)
        level_q = self._suggested_order(profile.acf, self.max_ma_order, threshold)
        difference_p = self._suggested_order(
            profile.differenced_pacf, self.max_ar_order, threshold
        )
        difference_q = self._suggested_order(
            profile.differenced_acf, self.max_ma_order, threshold
        )
        serial_memory = profile.ljung_box_pvalue < self.significance_level

        common_evidence = {
            "nobs": profile.nobs,
            "adf_pvalue": round(profile.adf_pvalue, 5),
            "kpss_pvalue": round(profile.kpss_pvalue, 5),
        }

        def add(
            spec: ModelSpec,
            *,
            rule: str,
            reason: str,
            evidence: Mapping[str, Any] | None = None,
        ) -> None:
            if spec in seen:
                return
            seen.add(spec)
            family = spec.family
            details = dict(evidence or {})
            if family not in available:
                decisions.append(
                    CandidateDecision(
                        "excluded",
                        "adapter_unavailable",
                        family,
                        spec,
                        f"adaptador {family} não está registrado no motor",
                        details,
                    )
                )
                return
            if len(candidates) >= self.max_candidates:
                decisions.append(
                    CandidateDecision(
                        "excluded",
                        "candidate_budget",
                        family,
                        spec,
                        f"orçamento de {self.max_candidates} candidatos atingido",
                        details,
                    )
                )
                return
            candidates.append(spec)
            decisions.append(
                CandidateDecision("included", rule, family, spec, reason, details)
            )

        def exclude(
            *,
            rule: str,
            family: str,
            spec: ModelSpec | None,
            reason: str,
            evidence: Mapping[str, Any] | None = None,
        ) -> None:
            decisions.append(
                CandidateDecision(
                    "excluded", rule, family, spec, reason, dict(evidence or {})
                )
            )

        # Benchmarks simples permanecem mesmo quando a caracterização é ambígua.
        add(
            ARIMASpec(order=(0, 0, 0), simulation_paths=self.simulation_paths),
            rule="mean_baseline",
            reason="baseline sem memória; mede se dinâmica adicional realmente ajuda",
            evidence={"stationarity": stationarity},
        )
        add(
            AR1Spec(),
            rule="transparent_ar1_baseline",
            reason="benchmark autoregressivo analítico e interpretável",
            evidence={"lag1_pacf": round(profile.pacf[1], 4)},
        )
        add(
            LocalLevelSpec(),
            rule="state_space_baseline",
            reason=(
                "nível latente é plausível diante de tendência/quebra/não estacionariedade"
                if stationarity != "stationary"
                or profile.summary["trend_strength"] >= 1
                or profile.change_improvement_ratio >= self.change_ratio_threshold
                else "benchmark de estado-espaço para nível lentamente variável"
            ),
            evidence={
                "trend_strength": round(profile.summary["trend_strength"], 4),
                "change_ratio": round(profile.change_improvement_ratio, 4),
            },
        )

        # Grade d=0: primeira ordem sempre; ordens maiores só com barras significativas.
        level_p = max(1, level_p if serial_memory else 1)
        level_q = max(1, level_q if serial_memory else 1)
        level_evidence = {
            **common_evidence,
            "pacf_order_signal": level_p,
            "acf_order_signal": level_q,
            "correlation_threshold": round(threshold, 4),
        }
        # Primeiro reservamos espaço para a grade mínima e para famílias especializadas.
        self._add_arima_grid(
            add,
            d=0,
            max_p=1,
            max_q=1,
            rule="level_acf_pacf",
            reason_prefix="ACF/PACF do nível propõem",
            evidence=level_evidence,
        )

        use_difference = self.differencing == "compare" or (
            self.differencing == "auto" and stationarity != "stationary"
        )
        if use_difference:
            difference_p = max(1, difference_p)
            difference_q = max(1, difference_q)
            difference_evidence = {
                **common_evidence,
                "stationarity": stationarity,
                "differenced_pacf_order_signal": difference_p,
                "differenced_acf_order_signal": difference_q,
            }
            self._add_arima_grid(
                add,
                d=1,
                max_p=1,
                max_q=1,
                rule="differenced_acf_pacf",
                reason_prefix="ADF/KPSS e ACF/PACF da diferença propõem",
                evidence=difference_evidence,
            )
        else:
            reason = (
                "diferenciação desabilitada pela política"
                if self.differencing == "never"
                else "ADF e KPSS são compatíveis com estacionariedade; d=1 foi podado"
            )
            exclude(
                rule="differencing_gate",
                family="arima",
                spec=ARIMASpec(
                    order=(0, 1, 0), simulation_paths=self.simulation_paths
                ),
                reason=reason,
                evidence=common_evidence,
            )

        self._consider_garch(
            profile, series_kind, available, add, exclude
        )
        self._consider_markov(profile, available, add, exclude)

        # Só depois das famílias especializadas usamos o restante do orçamento
        # para ordens 2 sugeridas pelas correlações.
        if level_p > 1 or level_q > 1:
            self._add_arima_grid(
                add,
                d=0,
                max_p=level_p,
                max_q=level_q,
                rule="level_acf_pacf",
                reason_prefix="ACF/PACF do nível propõem",
                evidence=level_evidence,
            )
        if use_difference and (difference_p > 1 or difference_q > 1):
            self._add_arima_grid(
                add,
                d=1,
                max_p=difference_p,
                max_q=difference_q,
                rule="differenced_acf_pacf",
                reason_prefix="ADF/KPSS e ACF/PACF da diferença propõem",
                evidence=difference_evidence,
            )

        summary = {
            "stationarity_assessment": stationarity,
            "correlation_threshold": float(threshold),
            "level_ar_order_signal": level_p,
            "level_ma_order_signal": level_q,
            "differenced_ar_order_signal": difference_p,
            "differenced_ma_order_signal": difference_q,
            "series_kind": series_kind,
            "candidate_count": len(candidates),
            "candidate_budget": self.max_candidates,
        }
        policy_snapshot = asdict(self)
        plan_payload = {
            "series_id": profile.series_id,
            "candidates": tuple(candidates),
            "decisions": tuple(decisions),
            "summary": summary,
            "policy": policy_snapshot,
        }
        return CandidatePlan(
            plan_id=content_id("candidate_plan", plan_payload),
            series_id=profile.series_id,
            candidates=tuple(candidates),
            decisions=tuple(decisions),
            summary=summary,
            policy=policy_snapshot,
        )

    def _stationarity_assessment(self, profile: SeriesProfile) -> str:
        if not (
            np.isfinite(profile.adf_pvalue) and np.isfinite(profile.kpss_pvalue)
        ):
            return "ambiguous"
        adf_rejects_unit_root = (
            profile.adf_pvalue < self.significance_level
        )
        kpss_does_not_reject = (
            profile.kpss_pvalue > self.significance_level
        )
        if adf_rejects_unit_root and kpss_does_not_reject:
            return "stationary"
        if not adf_rejects_unit_root and not kpss_does_not_reject:
            return "nonstationary"
        return "ambiguous"

    @staticmethod
    def _suggested_order(
        correlations: Sequence[float], maximum: int, threshold: float
    ) -> int:
        significant = [
            lag
            for lag in range(1, min(maximum, len(correlations) - 1) + 1)
            if abs(correlations[lag]) > threshold
        ]
        return max(significant, default=0)

    def _add_arima_grid(
        self,
        add,
        *,
        d: int,
        max_p: int,
        max_q: int,
        rule: str,
        reason_prefix: str,
        evidence: Mapping[str, Any],
    ) -> None:
        orders: list[tuple[int, int, int]] = [(1, d, 0), (0, d, 1), (1, d, 1)]
        if d == 1:
            orders.insert(0, (0, 1, 0))
        orders.extend((p, d, 0) for p in range(2, max_p + 1))
        orders.extend((0, d, q) for q in range(2, max_q + 1))
        orders.extend(
            (p, d, q)
            for p in range(1, max_p + 1)
            for q in range(1, max_q + 1)
            if p + q <= self.max_total_order
        )
        for order in dict.fromkeys(orders):
            add(
                ARIMASpec(order=order, simulation_paths=self.simulation_paths),
                rule=rule,
                reason=f"{reason_prefix} ARIMA{order}",
                evidence=evidence,
            )

    def _consider_garch(self, profile, series_kind, available, add, exclude) -> None:
        spec = GARCH11Spec(mean="constant", simulation_paths=max(500, self.simulation_paths))
        evidence = {
            "series_kind": series_kind,
            "arch_lm_pvalue": round(profile.arch_lm_pvalue, 5),
            "nobs": profile.nobs,
        }
        if self.garch == "never":
            exclude(rule="garch_gate", family="garch11", spec=spec,
                    reason="GARCH desabilitado pela política", evidence=evidence)
        elif self.garch == "auto" and series_kind != "returns":
            exclude(rule="garch_gate", family="garch11", spec=spec,
                    reason="GARCH automático exige SERIES_KIND='returns'", evidence=evidence)
        elif profile.nobs < self.minimum_garch_observations:
            exclude(rule="garch_gate", family="garch11", spec=spec,
                    reason=f"GARCH exige ao menos {self.minimum_garch_observations} observações",
                    evidence=evidence)
        elif self.garch == "auto" and (
            not np.isfinite(profile.arch_lm_pvalue)
            or profile.arch_lm_pvalue >= self.significance_level
        ):
            exclude(rule="garch_gate", family="garch11", spec=spec,
                    reason="ARCH-LM não indica volatilidade condicional", evidence=evidence)
        else:
            add(spec, rule="conditional_variance",
                reason="retornos e ARCH-LM sustentam testar variância GARCH",
                evidence=evidence)

    def _consider_markov(self, profile, available, add, exclude) -> None:
        spec = TwoRegimeGaussianSpec(max_iterations=200)
        evidence = {
            "change_ratio": round(profile.change_improvement_ratio, 5),
            "threshold": self.change_ratio_threshold,
            "nobs": profile.nobs,
            "free_parameters": 6,
            "observations_per_parameter": round(profile.nobs / 6, 2),
        }
        if self.markov_switching == "never":
            exclude(rule="markov_gate", family="markov_switching", spec=spec,
                    reason="Markov switching desabilitado pela política", evidence=evidence)
        elif profile.nobs < self.minimum_markov_observations:
            exclude(rule="markov_gate", family="markov_switching", spec=spec,
                    reason=f"Markov switching exige ao menos {self.minimum_markov_observations} observações",
                    evidence=evidence)
        elif self.markov_switching == "auto" and (
            profile.change_improvement_ratio < self.change_ratio_threshold
        ):
            exclude(rule="markov_gate", family="markov_switching", spec=spec,
                    reason="sinal de mudança estrutural abaixo do limiar", evidence=evidence)
        else:
            short_sample_note = (
                "; amostra curta: interpretar a hipótese de regimes como exploratória"
                if profile.nobs < 80
                else ""
            )
            add(spec, rule="regime_signal",
                reason=(
                    "sinal de quebra e razão observações/parâmetro sustentam testar dois regimes"
                    + short_sample_note
                ),
                evidence=evidence)
