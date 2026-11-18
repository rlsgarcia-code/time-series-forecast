"""Grafo pequeno para composição probabilística de séries previstas e conhecidas.

O módulo não ajusta modelos. Ele representa expressões, avalia matrizes de
trajetórias e estima a dependência usada pelo :class:`ForecastEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from .candidate_policy import CandidatePolicy
from .contracts import (
    BacktestConfig,
    EmpiricalDistribution,
    ModelSpec,
    PredictiveDistribution,
    TimeSeries,
    ValidationError,
    content_id,
    utc_now,
)


FloatArray = NDArray[np.float64]
VariableMode = Literal["forecast", "known", "constant", "scenario"]
DependencyMode = Literal["residual_copula", "independent"]


def _readonly_matrix(values: Any, *, name: str) -> FloatArray:
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 2 or np.any(~np.isfinite(result)):
        raise ValidationError(f"{name} deve ser uma matriz bidimensional finita")
    result.setflags(write=False)
    return result


def _source_signature(mode: VariableMode, source: Any) -> Mapping[str, Any]:
    if mode == "constant":
        return {"value": float(source)}
    if mode == "scenario":
        return {"scenarios": tuple(sorted(str(key) for key in source))}
    return {}


@dataclass(frozen=True, eq=False)
class ForecastVariable:
    """Uma folha ou expressão imutável do grafo probabilístico.

    ``source`` e a função privada existem somente em memória. O identificador
    estrutural usa nomes, modos, operadores e versões, nunca serializa o código.
    """

    kind: Literal["leaf", "operator", "transform"]
    name: str
    mode: VariableMode | None = None
    source: Any = field(default=None, repr=False, compare=False)
    vintages: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    operator: str | None = None
    operands: tuple["ForecastVariable", ...] = ()
    transform_version: str | None = None
    function: Callable[..., Any] | None = field(default=None, repr=False, compare=False)
    expression_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValidationError("nome da variável/transformação não pode ser vazio")
        if self.kind == "leaf":
            if self.mode not in {"forecast", "known", "constant", "scenario"}:
                raise ValidationError("modo de variável inválido")
            payload = {
                "kind": "leaf",
                "name": self.name,
                "mode": self.mode,
                **_source_signature(self.mode, self.source),
            }
        elif self.kind == "operator":
            if self.operator not in {"add", "sub", "mul", "div", "neg", "pow"}:
                raise ValidationError("operador de composição inválido")
            expected = 1 if self.operator == "neg" else 2
            if len(self.operands) != expected:
                raise ValidationError("quantidade inválida de operandos")
            payload = {
                "kind": "operator",
                "operator": self.operator,
                "operands": tuple(item.expression_id for item in self.operands),
            }
        elif self.kind == "transform":
            if self.function is None or not self.transform_version or not self.operands:
                raise ValidationError("transformação exige função, versão e entradas")
            payload = {
                "kind": "transform",
                "name": self.name,
                "version": self.transform_version,
                "operands": tuple(item.expression_id for item in self.operands),
            }
        else:
            raise ValidationError("tipo de nó inválido")
        object.__setattr__(self, "expression_id", content_id("expression", payload))

    @classmethod
    def leaf(
        cls,
        name: str,
        *,
        mode: VariableMode,
        source: Any,
        vintages: pd.DataFrame | None = None,
    ) -> "ForecastVariable":
        return cls("leaf", name, mode=mode, source=source, vintages=vintages)

    @classmethod
    def transform(
        cls,
        name: str,
        function: Callable[..., Any],
        operands: Sequence["ForecastVariable"],
        *,
        version: str,
    ) -> "ForecastVariable":
        return cls(
            "transform",
            name,
            operands=tuple(operands),
            transform_version=version,
            function=function,
        )

    def _binary(self, other: Any, operator: str) -> "ForecastVariable":
        right = _coerce_variable(other)
        return ForecastVariable(
            "operator",
            operator,
            operator=operator,
            operands=(self, right),
        )

    def __add__(self, other: Any) -> "ForecastVariable":
        return self._binary(other, "add")

    def __radd__(self, other: Any) -> "ForecastVariable":
        return _coerce_variable(other)._binary(self, "add")

    def __sub__(self, other: Any) -> "ForecastVariable":
        return self._binary(other, "sub")

    def __rsub__(self, other: Any) -> "ForecastVariable":
        return _coerce_variable(other)._binary(self, "sub")

    def __mul__(self, other: Any) -> "ForecastVariable":
        return self._binary(other, "mul")

    def __rmul__(self, other: Any) -> "ForecastVariable":
        return _coerce_variable(other)._binary(self, "mul")

    def __truediv__(self, other: Any) -> "ForecastVariable":
        return self._binary(other, "div")

    def __rtruediv__(self, other: Any) -> "ForecastVariable":
        return _coerce_variable(other)._binary(self, "div")

    def __neg__(self) -> "ForecastVariable":
        return ForecastVariable("operator", "neg", operator="neg", operands=(self,))

    def __pow__(self, exponent: Any) -> "ForecastVariable":
        if not np.isscalar(exponent) or not np.isfinite(float(exponent)):
            raise ValidationError("a potência deve usar expoente escalar finito")
        return self._binary(float(exponent), "pow")

    def to_payload(self) -> Mapping[str, Any]:
        """Referência segura: deliberadamente não revela a árvore nem a função."""
        return {"expression_id": self.expression_id, "name": self.name}


def _coerce_variable(value: Any) -> ForecastVariable:
    if isinstance(value, ForecastVariable):
        return value
    if np.isscalar(value) and np.isfinite(float(value)):
        scalar = float(value)
        return ForecastVariable.leaf(
            f"constant:{scalar.hex()}", mode="constant", source=scalar
        )
    raise ValidationError("operandos devem ser ForecastVariable ou escalares finitos")


@dataclass(frozen=True)
class CompositionPolicy:
    preselect_top_k: int = 2
    max_combinations: int = 32
    dependency: DependencyMode = "residual_copula"
    min_paired_errors: int = 12
    n_backtest_paths: int = 600
    confidence: float = 0.95
    coverage_range: tuple[float, float] | None = (0.80, 1.0)
    shrinkage: float | Literal["auto"] = "auto"
    output_weights: Mapping[str, float] = field(default_factory=dict)
    allow_noncausal_known: bool = False

    def __post_init__(self) -> None:
        if self.preselect_top_k < 1 or self.max_combinations < 1:
            raise ValidationError("limites de pré-seleção/composição devem ser positivos")
        if self.dependency not in {"residual_copula", "independent"}:
            raise ValidationError("dependency deve ser residual_copula ou independent")
        if self.min_paired_errors < 3 or self.n_backtest_paths < 100:
            raise ValidationError("dependência exige >=3 pares e backtest >=100 caminhos")
        if not 0 < self.confidence < 1:
            raise ValidationError("confidence deve estar em (0,1)")
        if self.coverage_range is not None:
            low, high = self.coverage_range
            if not 0 <= low <= high <= 1:
                raise ValidationError("coverage_range inválido")
        if self.shrinkage != "auto" and not 0 <= float(self.shrinkage) <= 1:
            raise ValidationError("shrinkage deve ser auto ou pertencer a [0,1]")
        if any(value <= 0 for value in self.output_weights.values()):
            raise ValidationError("pesos das saídas devem ser positivos")
        object.__setattr__(self, "output_weights", dict(self.output_weights))


@dataclass(frozen=True)
class DependencyDiagnostics:
    variable_names: tuple[str, ...]
    correlation: FloatArray
    paired_errors: int
    shrinkage: float
    fallback: bool
    reason: str
    realized_rank_correlation: FloatArray | None = None

    def __post_init__(self) -> None:
        correlation = _readonly_matrix(self.correlation, name="correlation")
        if correlation.shape != (len(self.variable_names), len(self.variable_names)):
            raise ValidationError("correlation incompatível com variable_names")
        object.__setattr__(self, "correlation", correlation)
        if self.realized_rank_correlation is not None:
            realized = _readonly_matrix(
                self.realized_rank_correlation, name="realized_rank_correlation"
            )
            object.__setattr__(self, "realized_rank_correlation", realized)

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "variable_names": self.variable_names,
            "correlation": self.correlation,
            "paired_errors": self.paired_errors,
            "shrinkage": self.shrinkage,
            "fallback": self.fallback,
            "reason": self.reason,
            "realized_rank_correlation": self.realized_rank_correlation,
        }


@dataclass(frozen=True)
class CompositionRun:
    composition_run_id: str
    expression_ids: Mapping[str, str]
    selected_model_runs: Mapping[str, str]
    selected_specs: Mapping[str, ModelSpec]
    input_revisions: Mapping[str, int]
    metrics: Mapping[str, float]
    scores: tuple[Mapping[str, Any], ...]
    dependency: DependencyDiagnostics
    causal: bool
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for field_name in (
            "expression_ids",
            "selected_model_runs",
            "selected_specs",
            "input_revisions",
            "metrics",
        ):
            object.__setattr__(self, field_name, dict(getattr(self, field_name)))

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "composition_run_id": self.composition_run_id,
            "expression_ids": self.expression_ids,
            "selected_model_runs": self.selected_model_runs,
            "input_revisions": self.input_revisions,
            "metrics": self.metrics,
            "dependency": self.dependency,
            "causal": self.causal,
            # selected_specs é necessário para auditoria, mas nenhuma árvore/função é persistida.
            "selected_specs": self.selected_specs,
        }


@dataclass(frozen=True)
class ComposedForecast:
    forecast_id: str
    composition_run_id: str
    targets: pd.DatetimeIndex
    distributions: Mapping[str, EmpiricalDistribution]
    expression_ids: Mapping[str, str]
    selected_model_runs: Mapping[str, str]
    input_revisions: Mapping[str, int]
    dependency: DependencyDiagnostics
    scenario: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        targets = pd.DatetimeIndex(self.targets).copy()
        if targets.tz is None or not targets.is_monotonic_increasing or not targets.is_unique:
            raise ValidationError("targets compostos devem ser ordenados, únicos e com timezone")
        distributions = dict(self.distributions)
        if not distributions or any(item.horizon != len(targets) for item in distributions.values()):
            raise ValidationError("distribuições compostas incompatíveis com targets")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "distributions", distributions)
        object.__setattr__(self, "expression_ids", dict(self.expression_ids))
        object.__setattr__(self, "selected_model_runs", dict(self.selected_model_runs))
        object.__setattr__(self, "input_revisions", dict(self.input_revisions))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def paths(self) -> Mapping[str, FloatArray]:
        return {name: distribution.paths for name, distribution in self.distributions.items()}


def normalize_outputs(
    outputs: ForecastVariable | Mapping[str, ForecastVariable],
) -> dict[str, ForecastVariable]:
    if isinstance(outputs, ForecastVariable):
        return {outputs.name: outputs}
    result = dict(outputs)
    if not result or any(not str(name).strip() for name in result):
        raise ValidationError("outputs deve conter ao menos uma saída nomeada")
    if any(not isinstance(item, ForecastVariable) for item in result.values()):
        raise ValidationError("todas as saídas devem ser ForecastVariable")
    return result


def collect_leaves(outputs: Mapping[str, ForecastVariable]) -> dict[str, ForecastVariable]:
    leaves: dict[str, ForecastVariable] = {}
    visited: set[str] = set()

    def visit(node: ForecastVariable) -> None:
        if node.expression_id in visited:
            return
        visited.add(node.expression_id)
        if node.kind == "leaf":
            previous = leaves.get(node.name)
            if previous is not None and previous.expression_id != node.expression_id:
                raise ValidationError(f"nome de variável reutilizado com outro contrato: {node.name}")
            leaves[node.name] = node
            return
        for operand in node.operands:
            visit(operand)

    for output in outputs.values():
        visit(output)
    return leaves


def evaluate_outputs(
    outputs: Mapping[str, ForecastVariable],
    leaf_values: Mapping[str, FloatArray],
) -> dict[str, FloatArray]:
    cache: dict[str, FloatArray] = {}
    expected_shape: tuple[int, int] | None = None
    for values in leaf_values.values():
        shape = np.asarray(values).shape
        if len(shape) == 2:
            expected_shape = shape
            break
    if expected_shape is None:
        raise ValidationError("ao menos uma folha deve fornecer matriz de trajetórias")

    def evaluate(node: ForecastVariable) -> FloatArray:
        if node.expression_id in cache:
            return cache[node.expression_id]
        if node.kind == "leaf":
            try:
                result = np.asarray(leaf_values[node.expression_id], dtype=float)
            except KeyError as error:
                raise ValidationError(f"valor ausente para a folha {node.name}") from error
        else:
            values = [evaluate(item) for item in node.operands]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                if node.kind == "transform":
                    assert node.function is not None
                    result = np.asarray(node.function(*values), dtype=float)
                elif node.operator == "add":
                    result = values[0] + values[1]
                elif node.operator == "sub":
                    result = values[0] - values[1]
                elif node.operator == "mul":
                    result = values[0] * values[1]
                elif node.operator == "div":
                    result = values[0] / values[1]
                elif node.operator == "neg":
                    result = -values[0]
                elif node.operator == "pow":
                    result = values[0] ** values[1]
                else:  # pragma: no cover - protegido pelo construtor
                    raise ValidationError("operador desconhecido")
        if result.shape != expected_shape or np.any(~np.isfinite(result)):
            raise ValidationError(
                f"nó {node.name} produziu forma/valores inválidos; esperado {expected_shape}"
            )
        readonly = np.asarray(result, dtype=float).copy()
        readonly.setflags(write=False)
        cache[node.expression_id] = readonly
        return readonly

    return {name: evaluate(node) for name, node in outputs.items()}


def estimate_dependency(
    errors: pd.DataFrame,
    variable_names: Sequence[str],
    *,
    mode: DependencyMode,
    min_paired_errors: int,
    shrinkage: float | Literal["auto"],
) -> DependencyDiagnostics:
    names = tuple(variable_names)
    dimension = len(names)
    identity = np.eye(dimension)
    if dimension <= 1:
        return DependencyDiagnostics(names, identity, len(errors), 1.0, False, "uma variável")
    paired = errors.reindex(columns=names).dropna()
    n_pairs = len(paired)
    if mode == "independent":
        return DependencyDiagnostics(names, identity, n_pairs, 1.0, True, "independência solicitada")
    if n_pairs < min_paired_errors:
        return DependencyDiagnostics(
            names,
            identity,
            n_pairs,
            1.0,
            True,
            f"{n_pairs} erros emparelhados; mínimo={min_paired_errors}",
        )
    sample = np.corrcoef(paired.to_numpy(dtype=float), rowvar=False)
    sample = np.atleast_2d(sample)
    sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(sample, 1.0)
    amount = (
        min(1.0, (dimension + 1) / max(n_pairs - 1, 1))
        if shrinkage == "auto"
        else float(shrinkage)
    )
    correlation = (1 - amount) * sample + amount * identity
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    correlation = (eigenvectors * np.clip(eigenvalues, 1e-8, None)) @ eigenvectors.T
    scale = np.sqrt(np.diag(correlation))
    correlation = correlation / np.outer(scale, scale)
    np.fill_diagonal(correlation, 1.0)
    return DependencyDiagnostics(
        names,
        correlation,
        n_pairs,
        amount,
        False,
        "cópula gaussiana estimada de erros PIT sincronizados",
    )


def impose_rank_dependence(
    paths: Mapping[str, FloatArray],
    diagnostics: DependencyDiagnostics,
    *,
    seed: int,
) -> tuple[dict[str, FloatArray], DependencyDiagnostics]:
    names = diagnostics.variable_names
    if not names:
        return dict(paths), diagnostics
    n_paths = np.asarray(paths[names[0]]).shape[0]
    if any(np.asarray(paths[name]).shape[0] != n_paths for name in names):
        raise ValidationError("todas as variáveis devem possuir o mesmo número de caminhos")
    rng = np.random.default_rng(seed)
    latent = rng.multivariate_normal(
        np.zeros(len(names)), diagnostics.correlation, size=n_paths
    )
    reordered: dict[str, FloatArray] = dict(paths)
    realized_scores = np.empty((n_paths, len(names)))
    for column, name in enumerate(names):
        original = np.asarray(paths[name], dtype=float)
        scale = np.std(original, axis=0, ddof=1)
        scale = np.where(scale > 1e-12, scale, 1.0)
        severity = np.mean((original - np.mean(original, axis=0)) / scale, axis=1)
        sorted_paths = original[np.argsort(severity)]
        latent_order = np.argsort(latent[:, column])
        aligned = np.empty_like(original)
        aligned[latent_order] = sorted_paths
        reordered[name] = aligned
        realized_scores[:, column] = np.mean(
            (aligned - np.mean(aligned, axis=0)) / scale, axis=1
        )
    if len(names) == 1:
        realized = np.ones((1, 1))
    else:
        ranks = pd.DataFrame(realized_scores).rank().to_numpy()
        realized = np.corrcoef(ranks, rowvar=False)
    updated = DependencyDiagnostics(
        names,
        diagnostics.correlation,
        diagnostics.paired_errors,
        diagnostics.shrinkage,
        diagnostics.fallback,
        diagnostics.reason,
        realized,
    )
    return reordered, updated


def _spec_id(spec: ModelSpec) -> str:
    return content_id("candidate", spec)


def _sample_crps(paths: FloatArray, actual: FloatArray, *, seed: int) -> FloatArray:
    rng = np.random.default_rng(seed)
    paired = paths[rng.permutation(paths.shape[0])]
    return np.mean(np.abs(paths - actual[None, :]), axis=0) - 0.5 * np.mean(
        np.abs(paths - paired), axis=0
    )


def _series_values(source: Any, targets: pd.DatetimeIndex, *, name: str) -> FloatArray:
    if np.isscalar(source):
        result = np.full(len(targets), float(source))
    elif isinstance(source, pd.Series):
        index = pd.DatetimeIndex(source.index)
        if index.tz is None:
            raise ValidationError(f"{name}: série conhecida deve ter timezone")
        selected = source.reindex(targets)
        if selected.isna().any():
            missing = [str(item) for item in selected.index[selected.isna()][:3]]
            raise ValidationError(f"{name}: valores conhecidos ausentes em {missing}")
        result = selected.to_numpy(dtype=float)
    else:
        raise ValidationError(f"{name}: fonte deve ser escalar ou pandas.Series")
    if np.any(~np.isfinite(result)):
        raise ValidationError(f"{name}: valores conhecidos devem ser finitos")
    return result


def _vintage_values(
    leaf: ForecastVariable,
    targets: pd.DatetimeIndex,
    *,
    as_of: pd.Timestamp,
) -> FloatArray:
    assert leaf.vintages is not None
    frame = leaf.vintages
    available = frame.loc[frame["as_of"] <= as_of].sort_values("as_of")
    latest = available.drop_duplicates("target", keep="last").set_index("target")["value"]
    selected = latest.reindex(targets)
    if selected.isna().any():
        missing = [str(item) for item in selected.index[selected.isna()][:3]]
        raise ValidationError(
            f"{leaf.name}: vintage indisponível no cutoff {as_of}; alvos {missing}"
        )
    return selected.to_numpy(dtype=float)


def _deterministic_source(leaf: ForecastVariable, scenario: str | None) -> Any:
    if leaf.mode != "scenario":
        return leaf.source
    if scenario is None:
        if "observed" in leaf.source:
            return leaf.source["observed"]
        if len(leaf.source) != 1:
            raise ValidationError(f"{leaf.name}: informe scenario")
        return next(iter(leaf.source.values()))
    try:
        return leaf.source[scenario]
    except KeyError as error:
        raise ValidationError(f"{leaf.name}: cenário desconhecido: {scenario}") from error


def materialize_leaves(
    leaves: Mapping[str, ForecastVariable],
    forecast_paths: Mapping[str, FloatArray],
    targets: pd.DatetimeIndex,
    *,
    n_paths: int,
    scenario: str | None,
    as_of: pd.Timestamp | None = None,
    use_vintages: bool = False,
) -> dict[str, FloatArray]:
    values: dict[str, FloatArray] = {}
    for leaf in leaves.values():
        if leaf.mode == "forecast":
            try:
                array = np.asarray(forecast_paths[leaf.name], dtype=float)
            except KeyError as error:
                raise ValidationError(f"caminhos ausentes para {leaf.name}") from error
        elif leaf.mode == "constant":
            array = np.full((n_paths, len(targets)), float(leaf.source))
        elif leaf.mode == "known" and use_vintages and leaf.vintages is not None:
            if as_of is None:
                raise ValidationError("as_of é obrigatório para vintages")
            vector = _vintage_values(leaf, targets, as_of=as_of)
            array = np.broadcast_to(vector, (n_paths, len(targets)))
        else:
            source = _deterministic_source(leaf, scenario)
            vector = _series_values(source, targets, name=leaf.name)
            array = np.broadcast_to(vector, (n_paths, len(targets)))
        readonly = np.asarray(array, dtype=float).copy()
        if readonly.shape != (n_paths, len(targets)) or np.any(~np.isfinite(readonly)):
            raise ValidationError(f"{leaf.name}: matriz de caminhos inválida")
        readonly.setflags(write=False)
        values[leaf.expression_id] = readonly
    return values


def _aligned_series(engine: Any, leaves: Sequence[ForecastVariable]) -> tuple[pd.DatetimeIndex, dict[str, TimeSeries]]:
    original = {leaf.name: engine.get_series(str(leaf.source)) for leaf in leaves}
    frequencies = {series.frequency for series in original.values()}
    cutoffs = {series.cutoff for series in original.values()}
    timezones = {str(series.timestamps.tz) for series in original.values()}
    if len(frequencies) != 1 or len(cutoffs) != 1 or len(timezones) != 1:
        raise ValidationError("variáveis previstas exigem frequência, timezone e cutoff comuns")
    common = next(iter(original.values())).timestamps
    for series in original.values():
        common = common.intersection(series.timestamps)
    if len(common) < 5:
        raise ValidationError("histórico comum insuficiente para composição")
    aligned: dict[str, TimeSeries] = {}
    for name, series in original.items():
        positions = series.timestamps.get_indexer(common)
        aligned[name] = TimeSeries(
            series.series_id,
            common,
            series.values[positions],
            series.frequency,
            series.metadata,
            series.revision,
        )
    return common, aligned


def _univariate_records(
    engine: Any,
    series: TimeSeries,
    specs: Sequence[ModelSpec],
    config: BacktestConfig,
    *,
    n_paths: int,
    seed_offset: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, int], FloatArray]]:
    records: dict[str, list[dict[str, Any]]] = {}
    path_cache: dict[tuple[str, int], FloatArray] = {}
    origins = list(
        range(config.initial_train_size, len(series.values) - config.horizon + 1, config.step)
    )
    if not origins:
        raise ValidationError("configuração não produz origens comuns de backtest")
    for candidate_number, spec in enumerate(specs):
        candidate_id = _spec_id(spec)
        candidate_records: list[dict[str, Any]] = []
        for origin_number, origin in enumerate(origins):
            train = TimeSeries(
                series.series_id,
                series.timestamps[:origin],
                series.values[:origin],
                series.frequency,
                series.metadata,
                revision=origin_number,
            )
            run = engine._make_run(train, spec)
            adapter = engine.models.get(spec.family)
            seed = config.seed + seed_offset + candidate_number * 100_000 + origin
            distribution: PredictiveDistribution = adapter.forecast(
                run, config.horizon, seed=seed
            )
            paths = distribution.sample(n_paths, seed=seed + 1)
            actual = series.values[origin : origin + config.horizon]
            targets = series.timestamps[origin : origin + config.horizon]
            alpha = 1 - config.confidence
            bounds = distribution.quantile([alpha / 2, 1 - alpha / 2])
            cdf = np.clip(distribution.cdf(actual), 1e-6, 1 - 1e-6)
            crps = _sample_crps(paths, actual, seed=seed + 2)
            path_cache[(candidate_id, origin)] = paths
            candidate_records.append(
                {
                    "origin": origin,
                    "cutoff": train.cutoff,
                    "targets": targets,
                    "actual": actual,
                    "mean": distribution.mean,
                    "lower": bounds[0],
                    "upper": bounds[1],
                    "crps": crps,
                    "pit_z": norm.ppf(cdf),
                }
            )
        records[candidate_id] = candidate_records
    return records, path_cache


def _aggregate_univariate(records: Sequence[Mapping[str, Any]]) -> Mapping[str, float]:
    actual = np.concatenate([item["actual"] for item in records])
    mean = np.concatenate([item["mean"] for item in records])
    lower = np.concatenate([item["lower"] for item in records])
    upper = np.concatenate([item["upper"] for item in records])
    crps = np.concatenate([item["crps"] for item in records])
    return {
        "mae": float(np.mean(np.abs(actual - mean))),
        "rmse": float(np.sqrt(np.mean((actual - mean) ** 2))),
        "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "mean_width": float(np.mean(upper - lower)),
        "crps": float(np.mean(crps)),
    }


def _pit_frame(
    names: Sequence[str],
    selected: Mapping[str, ModelSpec],
    records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    before: pd.Timestamp | None = None,
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for name in names:
        candidate_records = records[name][_spec_id(selected[name])]
        values: dict[pd.Timestamp, float] = {}
        for item in candidate_records:
            target = pd.Timestamp(item["targets"][0])
            if before is None or target <= before:
                values[target] = float(item["pit_z"][0])
        columns[name] = pd.Series(values, dtype=float)
    return pd.DataFrame(columns).sort_index()


def _actual_leaf_values(
    leaves: Mapping[str, ForecastVariable],
    aligned: Mapping[str, TimeSeries],
    targets: pd.DatetimeIndex,
    *,
    scenario: str | None = None,
) -> dict[str, FloatArray]:
    values: dict[str, FloatArray] = {}
    for leaf in leaves.values():
        if leaf.mode == "forecast":
            series = aligned[leaf.name]
            positions = series.timestamps.get_indexer(targets)
            if np.any(positions < 0):
                raise ValidationError(f"realizados ausentes para {leaf.name}")
            vector = series.values[positions]
        else:
            vector = _series_values(
                _deterministic_source(leaf, scenario), targets, name=leaf.name
            )
        values[leaf.expression_id] = np.asarray(vector, dtype=float)[None, :]
    return values


def train_expression(
    engine: Any,
    outputs_input: ForecastVariable | Mapping[str, ForecastVariable],
    *,
    backtest: BacktestConfig,
    candidate_policy: CandidatePolicy | None,
    policy: CompositionPolicy | None,
    promote_as: str | None,
) -> CompositionRun:
    outputs = normalize_outputs(outputs_input)
    leaves = collect_leaves(outputs)
    forecast_leaves = tuple(item for item in leaves.values() if item.mode == "forecast")
    if not forecast_leaves:
        raise ValidationError("backtest composto exige ao menos uma variável forecast")
    policy = policy or CompositionPolicy()
    candidate_policy = candidate_policy or CandidatePolicy()
    noncausal_known = [
        item.name
        for item in leaves.values()
        if item.mode == "known" and isinstance(item.source, pd.Series) and item.vintages is None
    ]
    scenario_inputs = [item.name for item in leaves.values() if item.mode == "scenario"]
    causal = not noncausal_known and not scenario_inputs
    if (noncausal_known or scenario_inputs) and not policy.allow_noncausal_known:
        affected = noncausal_known + scenario_inputs
        raise ValidationError(
            "backtest causal exige vintages para known e não avalia cenários retrospectivos: "
            + ", ".join(affected)
        )
    if promote_as and not causal:
        raise ValidationError("resultado não causal não pode ser promovido")

    common_index, aligned = _aligned_series(engine, forecast_leaves)
    if backtest.initial_train_size + backtest.horizon > len(common_index):
        raise ValidationError("backtest não cabe no histórico comum")

    plans: dict[str, Any] = {}
    records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    caches: dict[str, dict[tuple[str, int], FloatArray]] = {}
    preselected: dict[str, tuple[ModelSpec, ...]] = {}
    candidate_priority: dict[tuple[str, str], float] = {}
    for leaf_number, leaf in enumerate(forecast_leaves):
        profile = engine.characterize(str(leaf.source))
        plans[leaf.name] = engine.propose_candidates(
            str(leaf.source), policy=candidate_policy, profile=profile
        )
        leaf_records, leaf_cache = _univariate_records(
            engine,
            aligned[leaf.name],
            plans[leaf.name].candidates,
            backtest,
            n_paths=policy.n_backtest_paths,
            seed_offset=(leaf_number + 1) * 1_000_000,
        )
        records[leaf.name] = leaf_records
        caches[leaf.name] = leaf_cache
        scored = [
            (spec, _aggregate_univariate(leaf_records[_spec_id(spec)]))
            for spec in plans[leaf.name].candidates
        ]
        if policy.coverage_range is not None:
            low, high = policy.coverage_range
            eligible = [item for item in scored if low <= item[1]["coverage"] <= high]
            if eligible:
                scored = eligible
        scored.sort(key=lambda item: item[1]["crps"])
        selected = tuple(item[0] for item in scored[: policy.preselect_top_k])
        if not selected:
            raise ValidationError(f"nenhum candidato pré-selecionado para {leaf.name}")
        preselected[leaf.name] = selected
        marginal_scale = float(np.mean(np.abs(np.diff(aligned[leaf.name].values))))
        if not np.isfinite(marginal_scale) or marginal_scale <= 1e-12:
            marginal_scale = float(np.std(aligned[leaf.name].values, ddof=1))
        if not np.isfinite(marginal_scale) or marginal_scale <= 1e-12:
            marginal_scale = 1.0
        metrics_by_id = {
            _spec_id(spec): metrics for spec, metrics in scored
        }
        for spec in selected:
            candidate_priority[(leaf.name, _spec_id(spec))] = (
                metrics_by_id[_spec_id(spec)]["crps"] / marginal_scale
            )

    names = tuple(item.name for item in forecast_leaves)
    combinations = list(product(*(preselected[name] for name in names)))
    combinations.sort(
        key=lambda combo: sum(
            candidate_priority[(name, _spec_id(spec))]
            for name, spec in zip(names, combo)
        )
    )
    combinations = combinations[: policy.max_combinations]
    origins = list(
        range(backtest.initial_train_size, len(common_index) - backtest.horizon + 1, backtest.step)
    )
    combination_scores: list[dict[str, Any]] = []
    all_rows: dict[str, list[dict[str, Any]]] = {}

    for combination_number, combo in enumerate(combinations):
        selected = dict(zip(names, combo))
        combination_id = content_id(
            "combination", {name: spec for name, spec in selected.items()}
        )
        rows: list[dict[str, Any]] = []
        for origin in origins:
            cutoff = common_index[origin - 1]
            targets = common_index[origin : origin + backtest.horizon]
            raw_paths = {
                name: caches[name][(_spec_id(selected[name]), origin)] for name in names
            }
            prior_errors = _pit_frame(names, selected, records, before=cutoff)
            dependency = estimate_dependency(
                prior_errors,
                names,
                mode=policy.dependency,
                min_paired_errors=policy.min_paired_errors,
                shrinkage=policy.shrinkage,
            )
            joint_paths, _ = impose_rank_dependence(
                raw_paths,
                dependency,
                seed=backtest.seed + combination_number * 100_000 + origin,
            )
            leaf_paths = materialize_leaves(
                leaves,
                joint_paths,
                targets,
                n_paths=policy.n_backtest_paths,
                scenario=None,
                as_of=cutoff,
                use_vintages=True,
            )
            predicted = evaluate_outputs(outputs, leaf_paths)
            actual_leaf = _actual_leaf_values(leaves, aligned, targets)
            actual_outputs = evaluate_outputs(outputs, actual_leaf)
            alpha = 1 - policy.confidence
            for output_name, paths in predicted.items():
                actual = actual_outputs[output_name][0]
                mean = np.mean(paths, axis=0)
                bounds = np.quantile(paths, [alpha / 2, 1 - alpha / 2], axis=0)
                crps = _sample_crps(
                    paths,
                    actual,
                    seed=backtest.seed + combination_number * 1_000 + origin,
                )
                for step in range(backtest.horizon):
                    rows.append(
                        {
                            "output": output_name,
                            "origin": origin,
                            "horizon": step + 1,
                            "actual": float(actual[step]),
                            "mean": float(mean[step]),
                            "lower": float(bounds[0, step]),
                            "upper": float(bounds[1, step]),
                            "crps": float(crps[step]),
                        }
                    )
        by_output: dict[str, Mapping[str, float]] = {}
        normalized: dict[str, float] = {}
        for output_name in outputs:
            selected_rows = [item for item in rows if item["output"] == output_name]
            actual = np.array([item["actual"] for item in selected_rows])
            mean = np.array([item["mean"] for item in selected_rows])
            lower = np.array([item["lower"] for item in selected_rows])
            upper = np.array([item["upper"] for item in selected_rows])
            crps = np.array([item["crps"] for item in selected_rows])
            scale = float(np.mean(np.abs(np.diff(actual)))) if len(actual) > 1 else 0.0
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = float(np.std(actual, ddof=1)) if len(actual) > 1 else 1.0
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            by_output[output_name] = {
                "mae": float(np.mean(np.abs(actual - mean))),
                "rmse": float(np.sqrt(np.mean((actual - mean) ** 2))),
                "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
                "mean_width": float(np.mean(upper - lower)),
                "crps": float(np.mean(crps)),
                "scale": scale,
                "normalized_crps": float(np.mean(crps) / scale),
            }
            normalized[output_name] = float(np.mean(crps) / scale)
        weights = {
            name: float(policy.output_weights.get(name, 1.0)) for name in outputs
        }
        total_weight = sum(weights.values())
        aggregate_crps = sum(weights[name] * normalized[name] for name in outputs) / total_weight
        aggregate_coverage = sum(
            weights[name] * by_output[name]["coverage"] for name in outputs
        ) / total_weight
        score = {
            "combination_id": combination_id,
            "specifications": {name: repr(spec) for name, spec in selected.items()},
            "aggregate_normalized_crps": float(aggregate_crps),
            "aggregate_coverage": float(aggregate_coverage),
            "by_output": by_output,
        }
        combination_scores.append(score)
        all_rows[combination_id] = rows

    eligible_scores = combination_scores
    if policy.coverage_range is not None:
        low, high = policy.coverage_range
        eligible_scores = [
            item for item in combination_scores if low <= item["aggregate_coverage"] <= high
        ]
    if not eligible_scores:
        raise ValidationError("nenhuma combinação passou o guardrail de cobertura")
    winner = min(eligible_scores, key=lambda item: item["aggregate_normalized_crps"])
    winner_specs = {
        name: next(
            spec
            for spec in preselected[name]
            if repr(spec) == winner["specifications"][name]
        )
        for name in names
    }
    fitted = {
        name: engine.fit(str(leaves[name].source), winner_specs[name]) for name in names
    }
    final_errors = _pit_frame(names, winner_specs, records)
    dependency = estimate_dependency(
        final_errors,
        names,
        mode=policy.dependency,
        min_paired_errors=policy.min_paired_errors,
        shrinkage=policy.shrinkage,
    )
    expression_ids = {name: node.expression_id for name, node in outputs.items()}
    input_revisions = {
        name: engine.get_series(str(leaves[name].source)).revision for name in names
    }
    payload = {
        "expression_ids": expression_ids,
        "selected_model_runs": {name: run.model_run_id for name, run in fitted.items()},
        "input_revisions": input_revisions,
        "metrics": winner,
        "dependency": dependency,
        "causal": causal,
    }
    run = CompositionRun(
        content_id("composition", payload),
        expression_ids,
        {name: item.model_run_id for name, item in fitted.items()},
        winner_specs,
        input_revisions,
        {
            "normalized_crps": winner["aggregate_normalized_crps"],
            "coverage": winner["aggregate_coverage"],
        },
        tuple(combination_scores),
        dependency,
        causal,
    )
    if hasattr(engine.repository, "save_composition_run"):
        engine.repository.save_composition_run(run)
    if promote_as:
        if not hasattr(engine.repository, "set_composition_alias"):
            raise ValidationError("repositório não suporta aliases de composição")
        engine.repository.set_composition_alias(promote_as, run.composition_run_id)
    return run


def compose_outputs(
    engine: Any,
    outputs_input: ForecastVariable | Mapping[str, ForecastVariable],
    *,
    model_ref: str | CompositionRun,
    horizon: int,
    n_paths: int,
    seed: int,
    scenario: str | None,
    metadata: Mapping[str, Any] | None,
) -> ComposedForecast:
    if horizon < 1 or n_paths < 2:
        raise ValidationError("horizon deve ser positivo e n_paths >=2")
    outputs = normalize_outputs(outputs_input)
    leaves = collect_leaves(outputs)
    if isinstance(model_ref, CompositionRun):
        run = model_ref
    else:
        if not hasattr(engine.repository, "get_composition_run"):
            raise ValidationError("repositório não suporta composições persistidas")
        try:
            run = engine.repository.get_composition_run(model_ref)
        except (KeyError, LookupError):
            run_id = engine.repository.get_composition_alias(model_ref)
            if run_id is None:
                raise ValidationError(f"composição/alias não encontrado: {model_ref}")
            run = engine.repository.get_composition_run(run_id)
    expression_ids = {name: node.expression_id for name, node in outputs.items()}
    if expression_ids != dict(run.expression_ids):
        raise ValidationError("a expressão reconstruída não corresponde ao CompositionRun")
    forecast_leaves = tuple(item for item in leaves.values() if item.mode == "forecast")
    _, aligned = _aligned_series(engine, forecast_leaves)
    raw_paths: dict[str, FloatArray] = {}
    targets: pd.DatetimeIndex | None = None
    input_revisions: dict[str, int] = {}
    for leaf_number, leaf in enumerate(forecast_leaves):
        series = engine.get_series(str(leaf.source))
        expected_revision = run.input_revisions.get(leaf.name)
        if expected_revision != series.revision:
            raise ValidationError(f"entrada {leaf.name} mudou de revisão; refaça o treinamento")
        model_run = engine.repository.get_model_run(run.selected_model_runs[leaf.name])
        if model_run.series_revision != series.revision or model_run.cutoff != series.cutoff:
            raise ValidationError(f"modelo obsoleto para {leaf.name}")
        adapter = engine.models.get(model_run.spec.family)
        raw_paths[leaf.name] = adapter.forecast(
            model_run, horizon, seed=seed + leaf_number * 10_000
        ).sample(n_paths, seed=seed + leaf_number * 10_000 + 1)
        offset = pd.tseries.frequencies.to_offset(series.frequency)
        leaf_targets = pd.date_range(series.cutoff + offset, periods=horizon, freq=series.frequency)
        if targets is None:
            targets = leaf_targets
        elif not targets.equals(leaf_targets):
            raise ValidationError("targets incompatíveis entre variáveis")
        input_revisions[leaf.name] = series.revision
    assert targets is not None
    joint_paths, dependency = impose_rank_dependence(raw_paths, run.dependency, seed=seed + 99)
    leaf_paths = materialize_leaves(
        leaves,
        joint_paths,
        targets,
        n_paths=n_paths,
        scenario=scenario,
    )
    evaluated = evaluate_outputs(outputs, leaf_paths)
    distributions = {name: EmpiricalDistribution(paths) for name, paths in evaluated.items()}
    payload = {
        "composition_run_id": run.composition_run_id,
        "targets": targets,
        "expression_ids": expression_ids,
        "input_revisions": input_revisions,
        "scenario": scenario,
        "paths": {name: content_id("paths", paths) for name, paths in evaluated.items()},
        "metadata": metadata or {},
    }
    result = ComposedForecast(
        content_id("composed_forecast", payload),
        run.composition_run_id,
        targets,
        distributions,
        expression_ids,
        run.selected_model_runs,
        input_revisions,
        dependency,
        scenario,
        metadata or {},
    )
    if hasattr(engine.repository, "save_composed_forecast"):
        engine.repository.save_composed_forecast(result)
    return result


__all__ = [
    "ComposedForecast",
    "CompositionPolicy",
    "CompositionRun",
    "DependencyDiagnostics",
    "ForecastVariable",
    "compose_outputs",
    "train_expression",
]
