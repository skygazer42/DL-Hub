from __future__ import annotations

import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class SplitFedStrategy(FederatedStrategy):
    def __init__(self, *, cut_ratio: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cut_ratio = float(cut_ratio)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        cut = max(1, min(self.param_dim - 1, int(round(self.param_dim * self.cut_ratio))))
        client_front = state.server_params[:cut].unsqueeze(0) + state.raw_updates[:, :cut]
        server_back = state.server_params[cut:] + weighted_average(
            state.raw_updates[:, cut:], state.client_weights
        )
        server_params = torch.cat(
            [weighted_average(client_front, state.client_weights), server_back], dim=0
        )
        return {
            "server_params": server_params,
            "client_front": client_front,
            "server_back": server_back,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "splitfed_tiny": {"hidden": 8, "step_scale": 0.08, "cut_ratio": 0.5},
    "splitfed_small": {"hidden": 12, "step_scale": 0.12, "cut_ratio": 0.5},
    "splitfed_base": {"hidden": 16, "step_scale": 0.16, "cut_ratio": 0.5},
}


def build_splitfed_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "splitfed_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        SplitFedStrategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="splitfed",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_splitfed_strategy, "splitfed_tiny")
