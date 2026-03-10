from __future__ import annotations

import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class SecureAggStrategy(FederatedStrategy):
    def __init__(self, *, mask_scale: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mask_scale = float(mask_scale)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        gen = torch.Generator().manual_seed(int(seed) + 97)
        masks = self.mask_scale * torch.randn(self.num_clients, self.param_dim, generator=gen)
        masked_updates = state.raw_updates + masks
        recovered = masked_updates - masks
        server_params = state.server_params + weighted_average(recovered, state.client_weights)
        return {
            "server_params": server_params,
            "masked_updates": masked_updates,
            "recovered_updates": recovered,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "secureagg_tiny": {"hidden": 8, "step_scale": 0.08, "mask_scale": 0.05},
    "secureagg_small": {"hidden": 12, "step_scale": 0.12, "mask_scale": 0.08},
    "secureagg_base": {"hidden": 16, "step_scale": 0.16, "mask_scale": 0.1},
}


def build_secureagg_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "secureagg_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        SecureAggStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="secureagg",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_secureagg_strategy, "secureagg_tiny")
