from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy, weighted_average


class STCStrategy(FederatedStrategy):
    def __init__(self, *, sparsity: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sparsity = float(sparsity)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        k = max(1, int(round((1.0 - self.sparsity) * self.param_dim)))
        topk = torch.topk(state.raw_updates.abs(), k=k, dim=1).indices
        mask = torch.zeros_like(state.raw_updates)
        mask.scatter_(1, topk, 1.0)
        ternary = torch.sign(state.raw_updates) * mask
        server_params = state.server_params + weighted_average(ternary, state.client_weights)
        return {
            "server_params": server_params,
            "ternary_updates": ternary,
            "sparsity": torch.tensor(self.sparsity, dtype=torch.float32),
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "stc_tiny": {"hidden": 8, "step_scale": 0.08, "sparsity": 0.75},
    "stc_small": {"hidden": 12, "step_scale": 0.12, "sparsity": 0.85},
    "stc_base": {"hidden": 16, "step_scale": 0.16, "sparsity": 0.9},
}


def build_stc_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "stc_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        STCStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="stc",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_stc_strategy, "stc_tiny")
