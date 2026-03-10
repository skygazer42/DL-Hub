from __future__ import annotations

import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class MoonStrategy(FederatedStrategy):
    def __init__(self, *, temperature: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.temperature = float(temperature)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        global_bank = torch.stack(
            [
                state.global_repr,
                torch.roll(state.global_repr, shifts=1, dims=0),
                -state.global_repr,
            ],
            dim=0,
        )
        contrastive_logits = state.client_repr @ global_bank.T / max(self.temperature, 1e-6)
        alignment = contrastive_logits[:, :1]
        client_params = state.server_params.unsqueeze(0) + state.raw_updates + 0.02 * alignment
        server_params = weighted_average(client_params, state.client_weights)
        return {
            "server_params": server_params,
            "client_params": client_params,
            "contrastive_logits": contrastive_logits,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "moon_tiny": {"hidden": 8, "step_scale": 0.08, "temperature": 0.5},
    "moon_small": {"hidden": 12, "step_scale": 0.12, "temperature": 0.3},
    "moon_base": {"hidden": 16, "step_scale": 0.16, "temperature": 0.2},
}


def build_moon_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "moon_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        MoonStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="moon",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_moon_strategy, "moon_tiny")
