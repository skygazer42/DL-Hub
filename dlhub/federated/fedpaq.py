from __future__ import annotations

import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class FedPAQStrategy(FederatedStrategy):
    def __init__(self, *, quant_levels: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.quant_levels = int(quant_levels)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        scale = float(max(self.quant_levels - 1, 1))
        quantized = torch.round(state.raw_updates * scale) / scale
        server_params = state.server_params + weighted_average(quantized, state.client_weights)
        return {
            "server_params": server_params,
            "quantized_updates": quantized,
            "quant_levels": torch.tensor(self.quant_levels, dtype=torch.float32),
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "fedpaq_tiny": {"hidden": 8, "step_scale": 0.08, "quant_levels": 8},
    "fedpaq_small": {"hidden": 12, "step_scale": 0.12, "quant_levels": 16},
    "fedpaq_base": {"hidden": 16, "step_scale": 0.16, "quant_levels": 32},
}


def build_fedpaq_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "fedpaq_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        FedPAQStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="fedpaq",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_fedpaq_strategy, "fedpaq_tiny")
