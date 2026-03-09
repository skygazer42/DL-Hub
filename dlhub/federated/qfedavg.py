from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy


class QFedAvgStrategy(FederatedStrategy):
    def __init__(self, *, q: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.q = float(q)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        losses = state.raw_updates.square().mean(dim=1).clamp_min(1e-6)
        fair_weights = state.client_weights * losses.pow(self.q)
        fair_weights = fair_weights / fair_weights.sum().clamp_min(1e-8)
        aggregated = (fair_weights.unsqueeze(1) * state.raw_updates).sum(dim=0)
        server_params = state.server_params + aggregated
        return {
            "server_params": server_params,
            "fair_weights": fair_weights,
            "client_losses": losses,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "qfedavg_tiny": {"hidden": 8, "step_scale": 0.08, "q": 0.5},
    "qfedavg_small": {"hidden": 12, "step_scale": 0.12, "q": 1.0},
    "qfedavg_base": {"hidden": 16, "step_scale": 0.16, "q": 2.0},
}


def build_qfedavg_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "qfedavg_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        QFedAvgStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="qfedavg",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_qfedavg_strategy, "qfedavg_tiny")
