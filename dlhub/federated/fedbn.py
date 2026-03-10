from __future__ import annotations

import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class FedBNStrategy(FederatedStrategy):
    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        gen = torch.Generator().manual_seed(int(seed) + 23)
        local_bn = 0.1 * torch.randn(self.num_clients, self.hidden_dim, generator=gen)
        shared = weighted_average(state.raw_updates, state.client_weights)
        server_params = state.server_params + shared
        return {
            "server_params": server_params,
            "shared_update": shared,
            "local_bn_stats": local_bn,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "fedbn_tiny": {"hidden": 8, "step_scale": 0.08},
    "fedbn_small": {"hidden": 12, "step_scale": 0.12},
    "fedbn_base": {"hidden": 16, "step_scale": 0.16},
}


def build_fedbn_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "fedbn_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        FedBNStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="fedbn",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_fedbn_strategy, "fedbn_tiny")
