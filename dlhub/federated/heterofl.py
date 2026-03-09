from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy


class HeteroFLStrategy(FederatedStrategy):
    def __init__(self, *, width_floor: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.width_floor = float(width_floor)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        ratios = torch.linspace(self.width_floor, 1.0, self.num_clients, dtype=torch.float32)
        masked_updates = state.raw_updates * ratios.unsqueeze(1)
        server_params = state.server_params + masked_updates.mean(dim=0)
        return {
            "server_params": server_params,
            "client_width_ratio": ratios,
            "masked_updates": masked_updates,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "heterofl_tiny": {"hidden": 8, "step_scale": 0.08, "width_floor": 0.25},
    "heterofl_small": {"hidden": 12, "step_scale": 0.12, "width_floor": 0.5},
    "heterofl_base": {"hidden": 16, "step_scale": 0.16, "width_floor": 0.75},
}


def build_heterofl_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "heterofl_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        HeteroFLStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="heterofl",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_heterofl_strategy, "heterofl_tiny")
