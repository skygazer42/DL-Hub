from __future__ import annotations

import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class DPFedAvgStrategy(FederatedStrategy):
    def __init__(self, *, noise_std: float, clip_norm: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.noise_std = float(noise_std)
        self.clip_norm = float(clip_norm)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        gen = torch.Generator().manual_seed(int(seed) + 71)
        norms = state.raw_updates.norm(dim=1, keepdim=True).clamp_min(1e-6)
        clipped = state.raw_updates * (self.clip_norm / norms).clamp(max=1.0)
        noise = self.noise_std * torch.randn(self.param_dim, generator=gen)
        server_params = (
            state.server_params + weighted_average(clipped, state.client_weights) + noise
        )
        return {
            "server_params": server_params,
            "clipped_updates": clipped,
            "privacy_noise": noise,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "dp_fedavg_tiny": {"hidden": 8, "step_scale": 0.08, "noise_std": 0.01, "clip_norm": 0.25},
    "dp_fedavg_small": {"hidden": 12, "step_scale": 0.12, "noise_std": 0.02, "clip_norm": 0.35},
    "dp_fedavg_base": {"hidden": 16, "step_scale": 0.16, "noise_std": 0.03, "clip_norm": 0.45},
}


def build_dp_fedavg_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "dp_fedavg_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        DPFedAvgStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="dp_fedavg",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_dp_fedavg_strategy, "dp_fedavg_tiny")
