from __future__ import annotations

import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class PerFedAvgStrategy(FederatedStrategy):
    def __init__(self, *, meta_lr: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.meta_lr = float(meta_lr)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        personalized_params = state.server_params.unsqueeze(0) + state.raw_updates
        meta_gradient = weighted_average(
            personalized_params - state.server_params.unsqueeze(0), state.client_weights
        )
        server_params = state.server_params + self.meta_lr * meta_gradient
        return {
            "server_params": server_params,
            "meta_gradient": meta_gradient,
            "personalized_params": personalized_params,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "per_fedavg_tiny": {"hidden": 8, "step_scale": 0.08, "meta_lr": 0.3},
    "per_fedavg_small": {"hidden": 12, "step_scale": 0.12, "meta_lr": 0.4},
    "per_fedavg_base": {"hidden": 16, "step_scale": 0.16, "meta_lr": 0.5},
}


def build_per_fedavg_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "per_fedavg_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        PerFedAvgStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="per_fedavg",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_per_fedavg_strategy, "per_fedavg_tiny")
