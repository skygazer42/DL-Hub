from __future__ import annotations

import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class FedRepStrategy(FederatedStrategy):
    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        split = max(1, self.param_dim // 2)
        representation_update = weighted_average(state.raw_updates[:, :split], state.client_weights)
        personalized_heads = state.server_params[split:].unsqueeze(0) + state.raw_updates[:, split:]
        server_params = torch.cat(
            [state.server_params[:split] + representation_update, state.server_params[split:]],
            dim=0,
        )
        return {
            "server_params": server_params,
            "representation_update": representation_update,
            "personalized_heads": personalized_heads,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "fedrep_tiny": {"hidden": 8, "step_scale": 0.08},
    "fedrep_small": {"hidden": 12, "step_scale": 0.12},
    "fedrep_base": {"hidden": 16, "step_scale": 0.16},
}


def build_fedrep_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "fedrep_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        FedRepStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="fedrep",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_fedrep_strategy, "fedrep_tiny")
