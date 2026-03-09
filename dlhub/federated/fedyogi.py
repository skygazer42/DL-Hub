from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy, weighted_average


class FedYogiStrategy(FederatedStrategy):
    def __init__(self, *, beta1: float, beta2: float, eps: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        grad = weighted_average(state.raw_updates, state.client_weights)
        m = (1.0 - self.beta1) * grad
        v = (1.0 - self.beta2) * grad.square() * torch.sign(grad.square())
        server_params = state.server_params + m / (v.abs().sqrt() + self.eps)
        return {
            "server_params": server_params,
            "first_moment": m,
            "yogi_state": v,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "fedyogi_tiny": {"hidden": 8, "step_scale": 0.08, "beta1": 0.9, "beta2": 0.99, "eps": 1e-3},
    "fedyogi_small": {"hidden": 12, "step_scale": 0.12, "beta1": 0.9, "beta2": 0.995, "eps": 1e-4},
    "fedyogi_base": {"hidden": 16, "step_scale": 0.16, "beta1": 0.9, "beta2": 0.999, "eps": 1e-5},
}


def build_fedyogi_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "fedyogi_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        FedYogiStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="fedyogi",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_fedyogi_strategy, "fedyogi_tiny")
