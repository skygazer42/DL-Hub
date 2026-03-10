from __future__ import annotations

import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class FedProxStrategy(FederatedStrategy):
    def __init__(self, *, prox_mu: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prox_mu = float(prox_mu)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        client_params = state.server_params.unsqueeze(0) + state.raw_updates / (1.0 + self.prox_mu)
        prox_penalty = self.prox_mu * (
            (client_params - state.server_params.unsqueeze(0)) ** 2
        ).mean(dim=1)
        server_params = weighted_average(client_params, state.client_weights)
        return {
            "server_params": server_params,
            "client_params": client_params,
            "client_weights": state.client_weights,
            "prox_penalty": prox_penalty,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "fedprox_tiny": {"hidden": 8, "step_scale": 0.08, "prox_mu": 0.1},
    "fedprox_small": {"hidden": 12, "step_scale": 0.12, "prox_mu": 0.2},
    "fedprox_base": {"hidden": 16, "step_scale": 0.16, "prox_mu": 0.4},
}


def build_fedprox_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "fedprox_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        FedProxStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="fedprox",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_fedprox_strategy, "fedprox_tiny")
