from __future__ import annotations
import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class APFLStrategy(FederatedStrategy):
    def __init__(self, *, alpha: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = float(alpha)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        local_params = state.server_params.unsqueeze(0) + state.raw_updates
        global_update = weighted_average(state.raw_updates, state.client_weights)
        global_params = state.server_params + global_update
        mixed_params = self.alpha * local_params + (1.0 - self.alpha) * global_params.unsqueeze(0)
        return {
            "server_params": global_params,
            "mixed_params": mixed_params,
            "mixing_alpha": torch.tensor(self.alpha, dtype=torch.float32),
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "apfl_tiny": {"hidden": 8, "step_scale": 0.08, "alpha": 0.4},
    "apfl_small": {"hidden": 12, "step_scale": 0.12, "alpha": 0.5},
    "apfl_base": {"hidden": 16, "step_scale": 0.16, "alpha": 0.6},
}


def build_apfl_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "apfl_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        APFLStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="apfl",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_apfl_strategy, "apfl_tiny")
