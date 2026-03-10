from __future__ import annotations

import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class DPFedProxStrategy(FederatedStrategy):
    def __init__(self, *, noise_std: float, prox_mu: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.noise_std = float(noise_std)
        self.prox_mu = float(prox_mu)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        gen = torch.Generator().manual_seed(int(seed) + 73)
        adjusted = state.raw_updates / (1.0 + self.prox_mu)
        noise = self.noise_std * torch.randn(self.param_dim, generator=gen)
        server_params = (
            state.server_params + weighted_average(adjusted, state.client_weights) + noise
        )
        return {
            "server_params": server_params,
            "adjusted_updates": adjusted,
            "privacy_noise": noise,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "dp_fedprox_tiny": {"hidden": 8, "step_scale": 0.08, "noise_std": 0.01, "prox_mu": 0.1},
    "dp_fedprox_small": {"hidden": 12, "step_scale": 0.12, "noise_std": 0.02, "prox_mu": 0.2},
    "dp_fedprox_base": {"hidden": 16, "step_scale": 0.16, "noise_std": 0.03, "prox_mu": 0.4},
}


def build_dp_fedprox_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "dp_fedprox_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        DPFedProxStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="dp_fedprox",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_dp_fedprox_strategy, "dp_fedprox_tiny")
