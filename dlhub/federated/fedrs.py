from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy


class FedRSStrategy(FederatedStrategy):
    def __init__(self, *, reweight_strength: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.reweight_strength = float(reweight_strength)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        class_bias = torch.softmax(state.client_repr[:, : min(4, self.hidden_dim)], dim=-1)
        rarity = 1.0 / class_bias.mean(dim=0).clamp_min(1e-6)
        weight = rarity.mean() * self.reweight_strength
        balanced_updates = state.raw_updates * weight
        server_params = state.server_params + balanced_updates.mean(dim=0)
        return {
            "server_params": server_params,
            "class_bias": class_bias,
            "rarity_score": rarity,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "fedrs_tiny": {"hidden": 8, "step_scale": 0.08, "reweight_strength": 0.5},
    "fedrs_small": {"hidden": 12, "step_scale": 0.12, "reweight_strength": 0.75},
    "fedrs_base": {"hidden": 16, "step_scale": 0.16, "reweight_strength": 1.0},
}


def build_fedrs_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "fedrs_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        FedRSStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="fedrs",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_fedrs_strategy, "fedrs_tiny")
