from __future__ import annotations

import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class FedDynStrategy(FederatedStrategy):
    def __init__(self, *, alpha: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = float(alpha)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        dynamic_term = self.alpha * (
            state.raw_updates.mean(dim=0, keepdim=True) - state.raw_updates
        )
        adjusted = state.raw_updates + dynamic_term
        client_params = state.server_params.unsqueeze(0) + adjusted
        server_params = weighted_average(client_params, state.client_weights)
        return {
            "server_params": server_params,
            "client_params": client_params,
            "dynamic_term": dynamic_term,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "feddyn_tiny": {"hidden": 8, "step_scale": 0.08, "alpha": 0.1},
    "feddyn_small": {"hidden": 12, "step_scale": 0.12, "alpha": 0.2},
    "feddyn_base": {"hidden": 16, "step_scale": 0.16, "alpha": 0.35},
}


def build_feddyn_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "feddyn_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        FedDynStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="feddyn",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_feddyn_strategy, "feddyn_tiny")
