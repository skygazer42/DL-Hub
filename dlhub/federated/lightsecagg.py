from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy, weighted_average


class LightSecAggStrategy(FederatedStrategy):
    def __init__(self, *, sketch_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sketch_dim = int(sketch_dim)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        sketch = state.raw_updates[:, : self.sketch_dim]
        padded = torch.zeros_like(state.raw_updates)
        padded[:, : self.sketch_dim] = sketch
        server_params = state.server_params + weighted_average(padded, state.client_weights)
        return {
            "server_params": server_params,
            "secure_sketch": sketch,
            "reconstructed_updates": padded,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "lightsecagg_tiny": {"hidden": 8, "step_scale": 0.08, "sketch_dim": 4},
    "lightsecagg_small": {"hidden": 12, "step_scale": 0.12, "sketch_dim": 6},
    "lightsecagg_base": {"hidden": 16, "step_scale": 0.16, "sketch_dim": 8},
}


def build_lightsecagg_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "lightsecagg_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        LightSecAggStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="lightsecagg",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_lightsecagg_strategy, "lightsecagg_tiny")
