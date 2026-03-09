from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy


class FedRoDStrategy(FederatedStrategy):
    def __init__(self, *, distill_weight: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.distill_weight = float(distill_weight)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        global_branch = state.raw_updates.mean(dim=0)
        personalized_branch = state.raw_updates + self.distill_weight * global_branch.unsqueeze(0)
        server_params = state.server_params + global_branch
        return {
            "server_params": server_params,
            "global_branch": global_branch,
            "personalized_branch": personalized_branch,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "fedrod_tiny": {"hidden": 8, "step_scale": 0.08, "distill_weight": 0.15},
    "fedrod_small": {"hidden": 12, "step_scale": 0.12, "distill_weight": 0.25},
    "fedrod_base": {"hidden": 16, "step_scale": 0.16, "distill_weight": 0.35},
}


def build_fedrod_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "fedrod_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        FedRoDStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="fedrod",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_fedrod_strategy, "fedrod_tiny")
