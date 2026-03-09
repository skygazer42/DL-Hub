from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy


class TERMStrategy(FederatedStrategy):
    def __init__(self, *, tau: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tau = float(tau)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        client_loss = state.raw_updates.square().mean(dim=1)
        tilted_weights = torch.softmax(client_loss / max(self.tau, 1e-6), dim=0)
        tilted_update = (tilted_weights.unsqueeze(1) * state.raw_updates).sum(dim=0)
        server_params = state.server_params + tilted_update
        return {
            "server_params": server_params,
            "tilted_weights": tilted_weights,
            "client_loss": client_loss,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "term_tiny": {"hidden": 8, "step_scale": 0.08, "tau": 0.5},
    "term_small": {"hidden": 12, "step_scale": 0.12, "tau": 0.3},
    "term_base": {"hidden": 16, "step_scale": 0.16, "tau": 0.2},
}


def build_term_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "term_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        TERMStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="term",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_term_strategy, "term_tiny")
