from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ChinchillaConfig:
    tokens_per_parameter: float = 20.0
    flops_per_parameter_token: float = 6.0


@dataclass(frozen=True)
class ChinchillaPlan:
    parameters: int
    tokens: int
    compute_budget_flops: float
    tokens_per_parameter: float


class ChinchillaPlanner:
    def __init__(self, config: ChinchillaConfig | None = None) -> None:
        self.config = config or ChinchillaConfig()

    def training_flops(self, *, parameters: int, tokens: int) -> float:
        if int(parameters) <= 0:
            raise ValueError("parameters must be > 0")
        if int(tokens) <= 0:
            raise ValueError("tokens must be > 0")
        return (
            float(self.config.flops_per_parameter_token)
            * float(parameters)
            * float(tokens)
        )

    def plan_for_parameters(self, parameters: int) -> ChinchillaPlan:
        params = int(parameters)
        if params <= 0:
            raise ValueError("parameters must be > 0")
        tokens = int(round(params * float(self.config.tokens_per_parameter)))
        return ChinchillaPlan(
            parameters=params,
            tokens=tokens,
            compute_budget_flops=self.training_flops(parameters=params, tokens=tokens),
            tokens_per_parameter=float(self.config.tokens_per_parameter),
        )

    def optimal_parameters_for_compute(self, compute_budget_flops: float) -> int:
        compute = float(compute_budget_flops)
        if compute <= 0.0:
            raise ValueError("compute_budget_flops must be > 0")
        return int(
            round(
                math.sqrt(
                    compute
                    / (
                        float(self.config.flops_per_parameter_token)
                        * float(self.config.tokens_per_parameter)
                    )
                )
            )
        )

    def plan_for_compute(self, compute_budget_flops: float) -> ChinchillaPlan:
        params = self.optimal_parameters_for_compute(compute_budget_flops)
        return self.plan_for_parameters(params)


__all__ = [
    "ChinchillaConfig",
    "ChinchillaPlan",
    "ChinchillaPlanner",
]
