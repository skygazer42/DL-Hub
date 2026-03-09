from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy


class FedGKTStrategy(FederatedStrategy):
    def __init__(self, *, transfer_scale: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.transfer_scale = float(transfer_scale)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        logits = state.client_repr @ state.global_repr.unsqueeze(1)
        transferred = state.raw_updates + self.transfer_scale * logits
        server_params = state.server_params + transferred.mean(dim=0)
        return {
            "server_params": server_params,
            "transfer_logits": logits.squeeze(1),
            "student_updates": transferred,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "fedgkt_tiny": {"hidden": 8, "step_scale": 0.08, "transfer_scale": 0.05},
    "fedgkt_small": {"hidden": 12, "step_scale": 0.12, "transfer_scale": 0.08},
    "fedgkt_base": {"hidden": 16, "step_scale": 0.16, "transfer_scale": 0.12},
}


def build_fedgkt_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "fedgkt_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        FedGKTStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="fedgkt",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_fedgkt_strategy, "fedgkt_tiny")
