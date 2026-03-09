from __future__ import annotations

import torch

from ._common import FederatedStrategy, build_federated_strategy, smoke_test_strategy, weighted_average


class PFedMeStrategy(FederatedStrategy):
    def __init__(self, *, beta: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beta = float(beta)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        personalized_params = state.server_params.unsqueeze(0) + state.raw_updates
        server_avg = weighted_average(personalized_params, state.client_weights)
        server_params = (1.0 - self.beta) * state.server_params + self.beta * server_avg
        regularization = ((personalized_params - state.server_params.unsqueeze(0)) ** 2).mean(dim=1)
        return {
            "server_params": server_params,
            "personalized_params": personalized_params,
            "regularization": regularization,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "pfedme_tiny": {"hidden": 8, "step_scale": 0.08, "beta": 0.25},
    "pfedme_small": {"hidden": 12, "step_scale": 0.12, "beta": 0.35},
    "pfedme_base": {"hidden": 16, "step_scale": 0.16, "beta": 0.45},
}


def build_pfedme_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "pfedme_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        PFedMeStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="pfedme",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_pfedme_strategy, "pfedme_tiny")
