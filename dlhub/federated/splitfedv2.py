from __future__ import annotations

import torch

from ._common import (
    FederatedStrategy,
    build_federated_strategy,
    smoke_test_strategy,
    weighted_average,
)


class SplitFedV2Strategy(FederatedStrategy):
    def __init__(self, *, cut_ratio: float, relay_scale: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cut_ratio = float(cut_ratio)
        self.relay_scale = float(relay_scale)

    def simulate_round(self, *, seed: int = 0) -> dict[str, torch.Tensor]:
        state = self._sample_round_state(seed=seed)
        cut = max(1, min(self.param_dim - 1, int(round(self.param_dim * self.cut_ratio))))
        relay = self.relay_scale * state.raw_updates.mean(dim=0)
        client_front = state.server_params[:cut].unsqueeze(0) + state.raw_updates[:, :cut]
        server_back = state.server_params[cut:] + weighted_average(
            state.raw_updates[:, cut:], state.client_weights
        )
        server_params = torch.cat(
            [
                weighted_average(client_front, state.client_weights) + relay[:cut],
                server_back + relay[cut:],
            ],
            dim=0,
        )
        return {
            "server_params": server_params,
            "relay_state": relay,
            "client_front": client_front,
        }


_VARIANTS: dict[str, dict[str, float | int]] = {
    "splitfedv2_tiny": {"hidden": 8, "step_scale": 0.08, "cut_ratio": 0.5, "relay_scale": 0.1},
    "splitfedv2_small": {"hidden": 12, "step_scale": 0.12, "cut_ratio": 0.5, "relay_scale": 0.15},
    "splitfedv2_base": {"hidden": 16, "step_scale": 0.16, "cut_ratio": 0.5, "relay_scale": 0.2},
}


def build_splitfedv2_strategy(
    *,
    param_dim: int,
    num_clients: int,
    local_steps: int,
    variant: str = "splitfedv2_tiny",
    width_mult: float = 1.0,
):
    return build_federated_strategy(
        SplitFedV2Strategy,
        variants=_VARIANTS,
        param_dim=param_dim,
        num_clients=num_clients,
        local_steps=local_steps,
        variant=variant,
        width_mult=width_mult,
        family="splitfedv2",
    )


if __name__ == "__main__":
    smoke_test_strategy(build_splitfedv2_strategy, "splitfedv2_tiny")
