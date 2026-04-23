from __future__ import annotations
import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class DittoStrategy(FederatedStrategy):
    def __init__(self, *, lam: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lam = float(lam)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        global_update = weighted_average(state.raw_updates, state.client_weights)
        personalized = (
            state.server_params.unsqueeze(0)
            + state.raw_updates
            - self.lam * global_update.unsqueeze(0)
        )
        server_params = state.server_params + global_update
        return {
            "server_params": server_params,
            "personalized_params": personalized,
            "global_update": global_update,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "ditto_tiny": {"hidden": 8, "step_scale": 0.08, "lam": 0.1},
    "ditto_small": {"hidden": 12, "step_scale": 0.12, "lam": 0.2},
    "ditto_base": {"hidden": 16, "step_scale": 0.16, "lam": 0.35},
}


def build_ditto_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "ditto_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        DittoStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="ditto",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_ditto_strategy, "ditto_tiny")
