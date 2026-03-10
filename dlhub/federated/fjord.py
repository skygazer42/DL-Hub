from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy


class FjORDStrategy(FederatedStrategy):
    def __init__(self, *, dropout_floor: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dropout_floor = float(dropout_floor)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        gen = torch.Generator().manual_seed(int(seed) + 41)
        keep_prob = torch.linspace(self.dropout_floor, 1.0, self.num_clients, dtype=torch.float32)
        mask = (
            torch.rand(self.num_clients, self.param_dim, generator=gen) < keep_prob.unsqueeze(1)
        ).to(torch.float32)
        dropped_updates = state.raw_updates * mask
        server_params = state.server_params + dropped_updates.mean(dim=0)
        return {
            "server_params": server_params,
            "client_keep_prob": keep_prob,
            "drop_mask": mask,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "fjord_tiny": {"hidden": 8, "step_scale": 0.08, "dropout_floor": 0.4},
    "fjord_small": {"hidden": 12, "step_scale": 0.12, "dropout_floor": 0.55},
    "fjord_base": {"hidden": 16, "step_scale": 0.16, "dropout_floor": 0.7},
}


def build_fjord_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "fjord_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        FjORDStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="fjord",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_fjord_strategy, "fjord_tiny")
