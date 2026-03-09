from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy


class AFLStrategy(FederatedStrategy):
    def __init__(self, *, focus_power: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.focus_power = float(focus_power)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        client_risk = state.raw_updates.abs().mean(dim=1)
        risk_weights = torch.softmax(client_risk * self.focus_power, dim=0)
        aggregated = (risk_weights.unsqueeze(1) * state.raw_updates).sum(dim=0)
        server_params = state.server_params + aggregated
        return {
            "server_params": server_params,
            "risk_weights": risk_weights,
            "client_risk": client_risk,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "afl_tiny": {"hidden": 8, "step_scale": 0.08, "focus_power": 2.0},
    "afl_small": {"hidden": 12, "step_scale": 0.12, "focus_power": 3.0},
    "afl_base": {"hidden": 16, "step_scale": 0.16, "focus_power": 4.0},
}


def build_afl_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "afl_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        AFLStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="afl",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_afl_strategy, "afl_tiny")
