from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy


class FedAMPStrategy(FederatedStrategy):
    def __init__(self, *, attn_temp: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attn_temp = float(attn_temp)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        sim = state.client_repr @ state.client_repr.T / max(self.attn_temp, 1e-6)
        attention = torch.softmax(sim, dim=-1)
        mixed_updates = attention @ state.raw_updates
        server_params = state.server_params + mixed_updates.mean(dim=0)
        return {
            "server_params": server_params,
            "attention_matrix": attention,
            "mixed_updates": mixed_updates,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "fedamp_tiny": {"hidden": 8, "step_scale": 0.08, "attn_temp": 0.5},
    "fedamp_small": {"hidden": 12, "step_scale": 0.12, "attn_temp": 0.35},
    "fedamp_base": {"hidden": 16, "step_scale": 0.16, "attn_temp": 0.25},
}


def build_fedamp_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "fedamp_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        FedAMPStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="fedamp",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_fedamp_strategy, "fedamp_tiny")
