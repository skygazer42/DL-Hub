from __future__ import annotations

from dataclasses import dataclass, field


HELM_METRIC_CATEGORIES = (
    "accuracy",
    "calibration",
    "robustness",
    "fairness",
    "bias",
    "toxicity",
    "efficiency",
)


@dataclass(frozen=True)
class HELMScenario:
    name: str
    kind: str = "core"
    metrics: tuple[str, ...] = HELM_METRIC_CATEGORIES

    def __post_init__(self) -> None:
        if self.kind not in {"core", "targeted"}:
            raise ValueError("kind must be either 'core' or 'targeted'")
        unknown = [metric for metric in self.metrics if metric not in HELM_METRIC_CATEGORIES]
        if unknown:
            raise ValueError(f"unknown HELM metrics: {unknown}")


@dataclass(frozen=True)
class HELMRun:
    model_name: str
    scenario: HELMScenario
    metric_scores: dict[str, float]
    prompts: tuple[str, ...] = ()
    completions: tuple[str, ...] = ()


@dataclass(frozen=True)
class HELMScenarioCoverage:
    kind: str
    run_count: int
    measured_pairs: int
    possible_pairs: int
    coverage: float


@dataclass(frozen=True)
class HELMModelSummary:
    model_name: str
    coverage: float
    measured_pairs: int
    possible_pairs: int
    metric_averages: dict[str, float]
    macro_average: float
    core_runs: int
    targeted_runs: int


@dataclass(frozen=True)
class HELMReport:
    coverage: float
    measured_pairs: int
    possible_pairs: int
    metric_matrix: dict[str, dict[str, float | None]]
    model_metric_matrix: dict[str, dict[str, dict[str, float | None]]]
    scenario_kind_coverage: dict[str, HELMScenarioCoverage]
    leaderboard: tuple[HELMModelSummary, ...]
    prompt_logs: dict[str, tuple[str, ...]]
    completion_logs: dict[str, tuple[str, ...]]


@dataclass
class HELMEvaluator:
    metric_categories: tuple[str, ...] = HELM_METRIC_CATEGORIES
    core_scenario_count: int = 16
    targeted_scenario_count: int = 26
    raw_logs_enabled: bool = True
    _runs: list[HELMRun] = field(default_factory=list, init=False, repr=False)

    def evaluate(self, runs: list[HELMRun] | tuple[HELMRun, ...]) -> HELMReport:
        matrix: dict[str, dict[str, float | None]] = {}
        model_metric_matrix: dict[str, dict[str, dict[str, float | None]]] = {}
        prompt_logs: dict[str, tuple[str, ...]] = {}
        completion_logs: dict[str, tuple[str, ...]] = {}
        measured_pairs = 0
        possible_pairs = 0
        self._runs = list(runs)
        kind_totals = {
            "core": {"run_count": 0, "measured_pairs": 0, "possible_pairs": 0},
            "targeted": {"run_count": 0, "measured_pairs": 0, "possible_pairs": 0},
        }
        model_totals: dict[str, dict[str, int]] = {}
        model_scores: dict[str, dict[str, list[float]]] = {}

        for run in self._runs:
            scenario_name = str(run.scenario.name)
            scenario_metrics = tuple(run.scenario.metrics)
            model_name = str(run.model_name)
            kind = str(run.scenario.kind)
            possible_pairs += len(scenario_metrics)
            measured_pairs += sum(metric in run.metric_scores for metric in scenario_metrics)
            kind_totals[kind]["run_count"] += 1
            kind_totals[kind]["possible_pairs"] += len(scenario_metrics)
            kind_totals[kind]["measured_pairs"] += sum(
                metric in run.metric_scores for metric in scenario_metrics
            )

            row = {metric: None for metric in self.metric_categories}
            for metric in scenario_metrics:
                row[metric] = run.metric_scores.get(metric)
            matrix[scenario_name] = row
            model_metric_matrix.setdefault(model_name, {})[scenario_name] = dict(row)

            totals = model_totals.setdefault(
                model_name,
                {
                    "measured_pairs": 0,
                    "possible_pairs": 0,
                    "core_runs": 0,
                    "targeted_runs": 0,
                },
            )
            totals["possible_pairs"] += len(scenario_metrics)
            totals["measured_pairs"] += sum(metric in run.metric_scores for metric in scenario_metrics)
            totals[f"{kind}_runs"] += 1

            metric_lists = model_scores.setdefault(model_name, {})
            for metric in scenario_metrics:
                score = run.metric_scores.get(metric)
                if score is not None:
                    metric_lists.setdefault(metric, []).append(float(score))

            if self.raw_logs_enabled:
                prompt_logs[scenario_name] = tuple(run.prompts)
                completion_logs[scenario_name] = tuple(run.completions)

        coverage = 0.0 if possible_pairs == 0 else measured_pairs / possible_pairs
        scenario_kind_coverage = {
            kind: HELMScenarioCoverage(
                kind=kind,
                run_count=int(values["run_count"]),
                measured_pairs=int(values["measured_pairs"]),
                possible_pairs=int(values["possible_pairs"]),
                coverage=(
                    0.0
                    if int(values["possible_pairs"]) == 0
                    else int(values["measured_pairs"]) / int(values["possible_pairs"])
                ),
            )
            for kind, values in kind_totals.items()
        }
        leaderboard_rows: list[HELMModelSummary] = []
        for model_name, totals in model_totals.items():
            metric_averages = {
                metric: sum(scores) / len(scores)
                for metric, scores in model_scores.get(model_name, {}).items()
            }
            macro_average = (
                0.0
                if not metric_averages
                else sum(metric_averages.values()) / len(metric_averages)
            )
            model_possible_pairs = int(totals["possible_pairs"])
            model_measured_pairs = int(totals["measured_pairs"])
            leaderboard_rows.append(
                HELMModelSummary(
                    model_name=model_name,
                    coverage=(
                        0.0
                        if model_possible_pairs == 0
                        else model_measured_pairs / model_possible_pairs
                    ),
                    measured_pairs=model_measured_pairs,
                    possible_pairs=model_possible_pairs,
                    metric_averages=metric_averages,
                    macro_average=macro_average,
                    core_runs=int(totals["core_runs"]),
                    targeted_runs=int(totals["targeted_runs"]),
                )
            )
        leaderboard = tuple(
            sorted(
                leaderboard_rows,
                key=lambda row: (-row.macro_average, -row.coverage, row.model_name),
            )
        )
        return HELMReport(
            coverage=coverage,
            measured_pairs=measured_pairs,
            possible_pairs=possible_pairs,
            metric_matrix=matrix,
            model_metric_matrix=model_metric_matrix,
            scenario_kind_coverage=scenario_kind_coverage,
            leaderboard=leaderboard,
            prompt_logs=prompt_logs,
            completion_logs=completion_logs,
        )


__all__ = [
    "HELMEvaluator",
    "HELMModelSummary",
    "HELMReport",
    "HELMRun",
    "HELMScenario",
    "HELMScenarioCoverage",
    "HELM_METRIC_CATEGORIES",
]
