from __future__ import annotations

import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class FedGraphStrategy(FederatedStrategy):
    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        client_params = state.server_params.unsqueeze(0) + state.raw_updates
        server_params = weighted_average(client_params, state.client_weights)
        return {
            "server_params": server_params,
            "client_params": client_params,
            "client_weights": state.client_weights,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "fedgraph_tiny": {"hidden": 8, "step_scale": 0.08},
    "fedgraph_small": {"hidden": 12, "step_scale": 0.12},
    "fedgraph_base": {"hidden": 16, "step_scale": 0.16},
}


def build_fedgraph_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "fedgraph_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        FedGraphStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="fedgraph",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_fedgraph_strategy, "fedgraph_tiny")

