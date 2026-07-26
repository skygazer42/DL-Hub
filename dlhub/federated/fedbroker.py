from __future__ import annotations
import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class FedBrokerStrategy(FederatedStrategy):
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
    "fedbroker_tiny": {"hidden": 8, "step_scale": 0.08},
    "fedbroker_small": {"hidden": 12, "step_scale": 0.12},
    "fedbroker_base": {"hidden": 16, "step_scale": 0.16},
}


def build_fedbroker_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "fedbroker_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        FedBrokerStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="fedbroker",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_fedbroker_strategy, "fedbroker_tiny")
